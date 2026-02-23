#!/bin/bash
# Pi 5 Camera Docker

case "$1" in
  build)
    docker build -t cam .
    ;;
  start)
    pkill -f rpicam-vid; docker stop cam 2>/dev/null; sleep 1
    rpicam-vid -t 0 --width 1280 --height 720 --framerate 30 --codec h264 --inline --listen --nopreview -o tcp://0.0.0.0:5000 >/dev/null 2>&1 &
    sleep 2
    docker run -d --rm --name cam --network host -v $(pwd)/mlp_weights_txt:/app/mlp_weights_txt cam
    echo "Démarré - tcp://IP:8554"
    ;;
  stop)
    pkill -f rpicam-vid; docker stop cam 2>/dev/null
    echo "Arrêté"
    ;;
  view)
    gst-launch-1.0 tcpclientsrc host=192.168.0.120 port=8554 ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
    ;;
  logs)
    docker logs -f cam
    ;;
  all)
     $0 build && $0 start
    ;;
  *)
    echo "Usage: $0 {|build|start|stop|view|logs|all}"
    ;;
esac
