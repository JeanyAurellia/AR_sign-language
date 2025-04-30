from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import logging
import os
from werkzeug.utils import secure_filename
import time

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DEBUG_FOLDER = 'debug'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load BISINDO Model
try:
    model = load_model('model/sign_model.h5')
    logger.info("✅ Model loaded successfully")
    input_shape = model.input_shape[1:3]  # Expected input dimensions (e.g., 224x224)
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    model = None

# BISINDO Labels (Modify according to your model)
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'
}

# Gesture Mapping
GESTURES = {
    "FIST": "✊",
    "OPEN": "✋",
    "POINT": "☝️",
    "OK": "👌",
    "ROCK": "🤘",
    "UNKNOWN": "❓"
}

def recognize_gesture(hand_landmarks):
    """Recognize basic hand gestures from MediaPipe landmarks"""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    
    # Finger tip landmarks
    tips = {
        'thumb': landmarks[4],
        'index': landmarks[8],
        'middle': landmarks[12],
        'ring': landmarks[16],
        'pinky': landmarks[20]
    }
    
    # Calculate distances
    thumb_index_dist = np.linalg.norm(np.array(tips['thumb']) - np.array(tips['index']))
    index_middle_dist = np.linalg.norm(np.array(tips['index']) - np.array(tips['middle']))
    
    # Gesture recognition logic
    if thumb_index_dist < 0.05 and index_middle_dist < 0.05:
        return "FIST"
    elif thumb_index_dist < 0.05 and index_middle_dist > 0.1:
        return "OK"
    elif thumb_index_dist > 0.1 and index_middle_dist < 0.05:
        return "POINT"
    elif all(tips[finger][1] < landmarks[2][1] for finger in tips):  # All fingers extended
        return "OPEN"
    else:
        return "UNKNOWN"

def process_image(img_array):
    """Process image for hand detection and classification"""
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
        
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return None, None, "No hand detected"
        
        hand_landmarks = results.multi_hand_landmarks[0]
        gesture = recognize_gesture(hand_landmarks)
        gesture_symbol = GESTURES.get(gesture, GESTURES["UNKNOWN"])
        
        # Get bounding box
        h, w = img_array.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
        
        # Expand bounding box
        expand = 0.2
        x_min = max(0, x_min - (x_max - x_min) * expand)
        y_min = max(0, y_min - (y_max - y_min) * expand)
        x_max = min(w, x_max + (x_max - x_min) * expand)
        y_max = min(h, y_max + (y_max - y_min) * expand)
        
        # Crop and resize
        hand_crop = img_array[int(y_min):int(y_max), int(x_min):int(x_max)]
        if hand_crop.size == 0:
            return None, gesture_symbol, "Hand cropping failed"
        
        hand_resized = cv2.resize(hand_crop, input_shape)
        hand_normalized = hand_resized.astype('float32') / 255.0
        
        return np.expand_dims(hand_normalized, axis=0), gesture_symbol, None

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read image directly from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process image
        input_array, gesture, error = process_image(img_array)
        if error:
            return jsonify({'error': error}), 400
        
        # Predict
        preds = model.predict(input_array)
        pred_class = np.argmax(preds)
        confidence = float(preds[0][pred_class])
        
        logger.info(f"Prediction time: {time.time() - start_time:.2f}s")
        
        return jsonify({
            'class': LABELS.get(pred_class, 'Unknown'),
            'confidence': confidence,
            'gesture': gesture,
            'error': None
        })
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)