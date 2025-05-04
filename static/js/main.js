document.addEventListener('DOMContentLoaded', () => {
    const gestureDiv = document.getElementById('gestureResult');
  
    async function captureAndPredict() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();
  
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
  
        setInterval(async () => {
          canvas.width = 224;
          canvas.height = 224;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
          const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
          const formData = new FormData();
          formData.append('image', blob, 'frame.jpg');
  
          const response = await fetch('/predict', { method: 'POST', body: formData });
          const data = await response.json();
  
          if (!data.error) {
            gestureDiv.textContent = data.class;
  
            // Trigger animasi ulang biar tiap deteksi baru animasinya muncul
            gestureDiv.style.animation = 'none';
            gestureDiv.offsetHeight; // trigger reflow
            gestureDiv.style.animation = null;
  
          } else {
            gestureDiv.textContent = "-";
          }
  
        }, 1000);
      } catch (err) {
        gestureDiv.textContent = "Error kamera!";
      }
    }
  
    captureAndPredict();
  });
  