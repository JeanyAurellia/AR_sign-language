AFRAME.registerComponent('markerhandler', {
    init: function () {
      this.el.addEventListener('markerFound', () => {
        console.log('Marker ketemu!');
      });
      this.el.addEventListener('markerLost', () => {
        console.log('Marker ilang!');
      });
    }
  });
  