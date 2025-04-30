document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const resultDiv = document.getElementById('result');
    const gestureDiv = document.getElementById('gesture');
    const confidenceBar = document.getElementById('confidenceMeter');
    const loadingDiv = document.getElementById('loading');
    
    // App State
    let isProcessing = false;
    let stream = null;
    let lastUpdate = 0;
    const UPDATE_INTERVAL = 500; // Processing interval in ms
    
    // Initialize Camera
    async function initCamera() {
        try {
            loadingDiv.textContent = "Starting camera...";
            loadingDiv.style.display = 'block';
            
            // Get camera stream
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            video.srcObject = stream;
            
            // Wait for video to be ready
            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });
            
            // Set canvas dimensions
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Start processing loop
            loadingDiv.style.display = 'none';
            processFrame();
            
        } catch (err) {
            loadingDiv.textContent = `Camera Error: ${err.message}`;
            console.error("Camera initialization failed:", err);
        }
    }
    
    // Main Processing Loop
    function processFrame() {
        if (!stream) return;
        
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Throttle processing
        const now = Date.now();
        if (now - lastUpdate > UPDATE_INTERVAL && !isProcessing) {
            captureAndPredict();
            lastUpdate = now;
        }
        
        requestAnimationFrame(processFrame);
    }
    
    // Capture and Send for Prediction
    async function captureAndPredict() {
        isProcessing = true;
        
        try {
            // Define region of interest (center 70%)
            const roiSize = Math.min(canvas.width, canvas.height) * 0.7;
            const roiX = (canvas.width - roiSize) / 2;
            const roiY = (canvas.height - roiSize) / 2;
            
            // Create ROI canvas
            const roiCanvas = document.createElement('canvas');
            roiCanvas.width = roiSize;
            roiCanvas.height = roiSize;
            const roiCtx = roiCanvas.getContext('2d');
            roiCtx.drawImage(
                canvas, 
                roiX, roiY, roiSize, roiSize, 
                0, 0, roiSize, roiSize
            );
            
            // Convert to JPEG Blob
            const blob = await new Promise((resolve) => {
                roiCanvas.toBlob(resolve, 'image/jpeg', 0.9);
            });
            
            if (!blob) {
                throw new Error("Failed to capture frame");
            }
            
            // Prepare and send request
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update UI
            updateResults(data, roiX, roiY, roiSize);
            
        } catch (err) {
            console.error("Prediction error:", err);
            resultDiv.textContent = "Error: " + (err.message || "Processing failed");
            gestureDiv.textContent = '';
            confidenceBar.style.width = '0%';
        } finally {
            isProcessing = false;
        }
    }
    
    // Update UI with Results
    function updateResults(data, x, y, size) {
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
            gestureDiv.textContent = '';
            confidenceBar.style.width = '0%';
            return;
        }
        
        // Update prediction results
        const percent = Math.round(data.confidence * 100);
        resultDiv.textContent = `Huruf: ${data.class}`;
        gestureDiv.textContent = `Gesture: ${data.gesture}`;
        
        // Update confidence meter
        confidenceBar.style.width = `${percent}%`;
        confidenceBar.style.backgroundColor = getConfidenceColor(data.confidence);
        
        // Draw bounding box
        ctx.strokeStyle = getConfidenceColor(data.confidence);
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, size, size);
        
        // Draw label
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`${data.class} ${data.gesture}`, x + 5, y > 20 ? y - 5 : y + 20);
    }
    
    // Helper: Get color based on confidence
    function getConfidenceColor(confidence) {
        return confidence > 0.7 ? '#4CAF50' : 
               confidence > 0.4 ? '#FFC107' : '#F44336';
    }
    
    // Cleanup
    function cleanup() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }
    
    // Event Listeners
    window.addEventListener('beforeunload', cleanup);
    
    // Initialize App
    initCamera();
});