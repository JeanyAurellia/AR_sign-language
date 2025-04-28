from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model/sign_model.h5')
label_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Load default camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi tangan: crop area tengah untuk simplifikasi
        x, y, w, h = 200, 100, 224, 224
        hand = frame[y:y+h, x:x+w]
        hand_resized = cv2.resize(hand, (224, 224))
        hand_normalized = hand_resized / 255.0
        input_array = np.expand_dims(hand_normalized, axis=0)

        # Prediksi gesture
        prediction = model.predict(input_array)
        predicted_label = label_dict[np.argmax(prediction)]

        # Tambahkan kotak dan label pada frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
