from flask import Flask, render_template, Response
import cv2
import torch
import lgpio
import time
import numpy as np

app = Flask(__name__)

# Initialize the lgpio library
try:
    gpio_chip = lgpio.gpiochip_open(0)  # Open the GPIO chip
    if gpio_chip < 0:
        raise RuntimeError("Failed to open GPIO chip")
    print(f"GPIO chip opened successfully: {gpio_chip}")
except Exception as e:
    print(f"Error: {e}")
    gpio_chip = None

# Servo setup
servo_pin = 12  # GPIO pin connected to the servo
if gpio_chip is not None:
    lgpio.gpio_claim_output(gpio_chip, servo_pin)  # Claim the pin as an output
    print(f"Servo pin {servo_pin} claimed successfully")

# Initialize the current servo position
current_position = 90  # Neutral position (90 degrees)

# Define the step size for small movements
step_size = 10  # Adjust this value to control the movement amount

# Function to set servo angle
def set_servo_angle(angle):
    if gpio_chip is None:
        print("GPIO chip not available")
        return

    # Debug: Print the angle
    print(f"Angle: {angle}")

    # Convert angle to duty cycle (500-2500 microseconds)
    duty_cycle = 500 + (angle / 180) * 2000

    # Debug: Print the duty_cycle value
    print(f"Calculated duty_cycle: {duty_cycle}")

    # Ensure duty_cycle is within the valid range (0 to 1,000,000 microseconds)
    duty_cycle = max(0, min(duty_cycle, 1000000))

    # Debug: Print the final duty_cycle value
    print(f"Final duty_cycle: {duty_cycle}")

    # Debug: Print the lgpio.tx_pwm parameters
    print(f"Calling lgpio.tx_pwm with: gpio_chip={gpio_chip}, servo_pin={servo_pin}, frequency=50, duty_cycle={int(duty_cycle)}")

    # Set PWM signal
    lgpio.tx_pwm(gpio_chip, servo_pin, 50, int(duty_cycle))  # 50 Hz, duty_cycle in microseconds

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
    current_position = max(0, current_position - step_size)  # Move left
    print(f"Moving left to {current_position} degrees")
    set_servo_angle(current_position)
    return f"Moved Left to {current_position}"

@app.route('/move_right')
def move_right():
    global current_position
    current_position = min(180, current_position + step_size)  # Move right
    print(f"Moving right to {current_position} degrees")
    set_servo_angle(current_position)
    return f"Moved Right to {current_position}"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        # Cleanup
        if gpio_chip is not None:
            lgpio.gpiochip_close(gpio_chip)
