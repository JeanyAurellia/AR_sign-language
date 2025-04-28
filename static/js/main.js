const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultText = document.getElementById('result');
const context = canvas.getContext('2d');

// Akses kamera
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        alert("Camera access denied or not found: " + err);
    });

// Kirim frame tiap 1 detik ke Flask
setInterval(() => {
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ image: dataURL }),
            headers: { 'Content-Type': 'application/json' }
        })
        .then(res => res.json())
        .then(data => {
            resultText.innerText = `Predicted Sign: ${data.class} (Confidence: ${data.confidence.toFixed(2)})`;
        })
        .catch(err => {
            resultText.innerText = 'Error predicting gesture';
            console.error(err);
        });
    }
}, 1000);
