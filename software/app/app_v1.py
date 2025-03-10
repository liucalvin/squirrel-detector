from flask import Flask, render_template, Response
import cv2
from gpiozero import Servo
from gpiozero.pins.rpigpio import RPiGPIOFactory
from gpiozero import Device
from time import sleep

app = Flask(__name__)

# Servo setup
servo_pin = 12  # GPIO pin connected to the servo
servo = Servo(servo_pin, initial_value=0, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000, frame_width=1/100)

# Initialize the current servo position
current_position = 0  # Neutral position (0)

# Define the step size for small movements
step_size = 0.2  # Adjust this value to control the movement amount

# Define a deadband to filter small changes
deadband = 0.05  # Ignore changes smaller than this value

# Camera setup
camera = cv2.VideoCapture(4)  # Use 0 for the default camera

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
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
    new_position = max(-1, current_position - step_size)
    if abs(new_position - current_position) >= deadband:
        current_position = new_position
        servo.value = current_position
        sleep(0.1)  # Add a small delay
    return f"Moved Left to {current_position}"

@app.route('/move_right')
def move_right():
    global current_position
    new_position = min(1, current_position + step_size)
    if abs(new_position - current_position) >= deadband:
        current_position = new_position
        servo.value = current_position
        sleep(0.1)  # Add a small delay
    return f"Moved Right to {current_position}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)