from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import lgpio
import time
import numpy as np

app = Flask(__name__)

# Open the GPIO chip
gpio_chip = lgpio.gpiochip_open(0)
if gpio_chip < 0:
    raise RuntimeError("Failed to open GPIO chip")

# Servo setup
servo_pin = 12  # GPIO pin connected to the servo
lgpio.gpio_claim_output(gpio_chip, servo_pin)  # Claim the pin as an output

# Initialize the current servo position
current_position = 90  # Neutral position (90 degrees)

# Define the step size for small movements
step_size = 10  # Adjust this value to control the movement amount

# Camera setup
camera = cv2.VideoCapture(4)  # Use 4 for the camera

# Load YOLOv11 model using Ultralytics
model = YOLO("app/yolo11n.pt")  # Replace with the path to your YOLOv11 model

# Load class names from coco.names
with open("app/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

def set_servo_angle(angle):
    # Convert angle to duty cycle (500-2500 microseconds)
    duty_cycle = 500 + (angle / 180) * 2000

    # Ensure duty_cycle is within the valid range (0 to 1,000,000 microseconds)
    duty_cycle = max(0, min(duty_cycle, 1000000))

    # Set PWM signal
    lgpio.tx_pwm(gpio_chip, servo_pin, 50, int(duty_cycle))  # 50 Hz, duty_cycle in microseconds

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
                    break

            # Enable or disable the servo based on person detection
            if person_detected:
                set_servo_angle(current_position)  # Enable the servo
            else:
                lgpio.tx_pwm(gpio_chip, servo_pin, 50, 0)  # Disable the servo

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
    current_position = max(0, current_position - step_size)  # Move left
    set_servo_angle(current_position)
    return f"Moved Left to {current_position}"

@app.route('/move_right')
def move_right():
    global current_position
    current_position = min(180, current_position + step_size)  # Move right
    set_servo_angle(current_position)
    return f"Moved Right to {current_position}"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        # Cleanup
        lgpio.gpiochip_close(gpio_chip)
