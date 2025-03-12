from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
import numpy as np

app = Flask(__name__)

# Servo setup
servo_pin = 13  # GPIO pin connected to the servo
servo = Servo(servo_pin, initial_value=None)  # Start with the servo disabled

# Initialize the current servo position
current_position = 0  # Neutral position (0)

# Define the step size for small movements
step_size = 0.01  # Adjust this value to control the movement speed

# Camera setup
camera = cv2.VideoCapture(4)  # Use 4 for the camera
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the frame
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the frame
frame_center_x = frame_width // 2  # Center of the frame (horizontal)

# Load YOLOv11 model using Ultralytics
model = YOLO("yolo11n.pt")  # Replace with the path to your YOLOv11 model

# Load class names from coco.names
with open("coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

def move_servo_to_center_person(person_center_x):
    global current_position

    # Calculate the horizontal distance between the person's center and the frame's center
    distance = person_center_x - frame_center_x

    # Move the servo to minimize the distance
    if distance > 50:  # Person is to the right of the center
        current_position = min(1, current_position + step_size)  # Move right
    elif distance < -50:  # Person is to the left of the center
        current_position = max(-1, current_position - step_size)  # Move left

    # Update the servo position
    servo.value = current_position

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model(frame)  # Run inference
            detections = results[0].boxes.data.cpu().numpy()  # Get detections

            # Check if a person is detected
            person_detected = False
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                if class_names[int(cls)] == "person":
                    person_detected = True

                    # Calculate the center of the bounding box
                    person_center_x = (x1 + x2) // 2

                    # Move the servo to center the person
                    move_servo_to_center_person(person_center_x)

                    # Draw bounding box and center point
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(person_center_x), int((y1 + y2) // 2)), 5, (0, 0, 255), -1)
                    break

            # Draw the center of the frame
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 0, 0), 2)

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

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        # Cleanup
        servo.value = None  # Disable the servo
