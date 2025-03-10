from flask import Flask, render_template, Response
import cv2
import torch
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
from time import sleep
import numpy as np

app = Flask(__name__)

# Servo setup
servo_pin = 12  # GPIO pin connected to the servo
servo = Servo(servo_pin, initial_value=None)  # Start with the servo disabled

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

# Load class names from coco.names
with open("app/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model(frame)  # Run inference
            detections = results.xyxy[0].numpy()  # Get detections

            # Check if a person is detected
            person_detected = False
            for detection in detections:
                cls = int(detection[5])
                if class_names[cls] == "person":
                    person_detected = True
                    break

            # Enable or disable the servo based on person detection
            if person_detected:
                servo.value = current_position  # Enable the servo
            else:
                servo.value = None  # Disable the servo

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
    current_position = max(-1, current_position - step_size)  # Move left
    if servo.value is not None:  # Only move if the servo is enabled
        servo.value = current_position
    return f"Moved Left to {current_position}"

@app.route('/move_right')
def move_right():
    global current_position
    current_position = min(1, current_position + step_size)  # Move right
    if servo.value is not None:  # Only move if the servo is enabled
        servo.value = current_position
    return f"Moved Right to {current_position}"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        # Cleanup
        servo.value = None  # Disable the servo
