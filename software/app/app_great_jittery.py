from flask import Flask, render_template, Response
import cv2
import torch
from gpiozero import Servo
from time import sleep
import numpy as np

app = Flask(__name__)

# Servo setup
servo_pin = 12  # GPIO pin connected to the servo
servo = Servo(servo_pin)

# Initialize the current servo position
current_position = 0  # Neutral position (0)

# Define the step size for small movements
step_size = 0.2  # Adjust this value to control the movement amount

# Define a deadband to filter small changes
deadband = 0.05  # Ignore changes smaller than this value

# Camera setup
camera = cv2.VideoCapture(4)  # Use 4 for the camera

# Load YOLOv11n model using PyTorch
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Use 'yolov5n' as a placeholder for YOLOv11n
model.eval()  # Set the model to evaluation mode

# Define class names (update this list with your model's class names)
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
               "hair drier", "toothbrush"]

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model(frame)  # Run inference
            detections = results.xyxy[0].numpy()  # Get detections

            # Draw bounding boxes on the frame
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                label = f"{class_names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/move_left')
def move_left():
    global current_position
    new_position = min(1, current_position + step_size)  # Adjusted for flipped motor
    if abs(new_position - current_position) >= deadband:
        current_position = new_position
        servo.value = current_position
        sleep(0.1)  # Add a small delay
    return f"Moved Left to {current_position}"

@app.route('/move_right')
def move_right():
    global current_position
    new_position = max(-1, current_position - step_size)  # Adjusted for flipped motor
    if abs(new_position - current_position) >= deadband:
        current_position = new_position
        servo.value = current_position
        sleep(0.1)  # Add a small delay
    return f"Moved Right to {current_position}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
