document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const resultDiv = document.getElementById('result');
    const gestureDiv = document.getElementById('gesture');
    const confidenceBar = document.getElementById('confidenceMeter');
    const loadingDiv = document.getElementById('loading');

    let isProcessing = false;
    let stream = null;
    let lastUpdate = 0;
    const UPDATE_INTERVAL = 800; // dalam ms

    async function initCamera() {
        try {
            loadingDiv.textContent = "Menghidupkan kamera...";
            loadingDiv.style.display = 'block';

            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false
            });

            video.srcObject = stream;

            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            loadingDiv.style.display = 'none';
            processFrame();

        } catch (err) {
            loadingDiv.textContent = `Gagal mengakses kamera: ${err.message}`;
            console.error("Camera initialization failed:", err);
        }
    }

    function processFrame() {
        if (!stream) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Tambahkan kotak panduan ROI
        const roiSize = Math.min(canvas.width, canvas.height) * 0.7;
        const roiX = (canvas.width - roiSize) / 2;
        const roiY = (canvas.height - roiSize) / 2;
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(roiX, roiY, roiSize, roiSize);

        const now = Date.now();
        if (now - lastUpdate > UPDATE_INTERVAL && !isProcessing) {
            captureAndPredict(roiX, roiY, roiSize);
            lastUpdate = now;
        }

        requestAnimationFrame(processFrame);
    }

    async function captureAndPredict(roiX, roiY, roiSize) {
        isProcessing = true;

        try {
            const roiCanvas = document.createElement('canvas');
            roiCanvas.width = roiSize;
            roiCanvas.height = roiSize;
            const roiCtx = roiCanvas.getContext('2d');
            roiCtx.drawImage(canvas, roiX, roiY, roiSize, roiSize, 0, 0, roiSize, roiSize);

            const blob = await new Promise((resolve) => {
                roiCanvas.toBlob(resolve, 'image/jpeg', 0.9);
            });

            if (!blob) throw new Error("Gagal mengambil gambar");

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
            updateResults(data, roiX, roiY, roiSize);

        } catch (err) {
            console.error("Prediction error:", err);
            resultDiv.textContent = "❌ " + (err.message || "Gagal memproses");
            gestureDiv.textContent = '';
            confidenceBar.style.width = '0%';
        } finally {
            isProcessing = false;
        }
    }

    function updateResults(data, x, y, size) {
        if (data.error) {
            resultDiv.textContent = `⚠️ ${data.error}`;
            gestureDiv.textContent = '';
            confidenceBar.style.width = '0%';
            return;
        }

        const percent = Math.round(data.confidence * 100);
        resultDiv.textContent = `Huruf: ${data.class}`;
        gestureDiv.textContent = `Gesture: ${data.gesture}`;

        confidenceBar.style.width = `${percent}%`;
        confidenceBar.style.backgroundColor = getConfidenceColor(data.confidence);

        ctx.strokeStyle = getConfidenceColor(data.confidence);
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, size, size);

        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`${data.class} ${data.gesture}`, x + 5, y > 20 ? y - 5 : y + 20);
    }

    function getConfidenceColor(confidence) {
        return confidence > 0.7 ? '#4CAF50' :
               confidence > 0.4 ? '#FFC107' : '#F44336';
    }

    function cleanup() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    }

    window.addEventListener('beforeunload', cleanup);
    initCamera();
});
