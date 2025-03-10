from flask import Flask, Response
import cv2

RGB_STREAM_CHANNEL = 4

app = Flask(__name__)

def generate():
    cap = cv2.VideoCapture(RGB_STREAM_CHANNEL, cv2.CAP_V4L2)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()

@app.route('/video')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
