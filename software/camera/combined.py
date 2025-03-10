import cv2
import logging
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO
from flask import Flask, render_template, Response
import threading

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask Setup
app = Flask(__name__)

# Yaw Motor Setup (Left/Right)
SERVO_YAW_PIN = 12
yaw_motor = PWMOutputDevice(SERVO_YAW_PIN, frequency=50)  # 50 Hz PWM

# Yaw motor limits
YAW_MIN = -1.0   # Leftmost position
YAW_MAX = 1.0    # Rightmost position

# Define the duty cycle range for PWM (servo-specific)
MIN_DUTY = 0.05
MAX_DUTY = 0.11

# PID Controller Parameters (For Smooth Movement)
Kp = 0.1   # Proportional gain (adjusts response)
Ki = 0.05  # Integral gain (helps with drift)
Kd = 0.02  # Derivative gain (reduces overshoot)

MAX_YAW_SPEED = 0.05  # Max speed change per update
previous_error = 0
integral = 0

# LED setup (to turn on when a person is detected)
led = LED(17)

# Video capture setup
FRAME_WIDTH = 720
FRAME_HEIGHT = 480
model = YOLO("yolov5s.pt")
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Shared variable for current yaw position
yaw_position = 0  # Start at the center

def map_range(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another while keeping it within limits."""
    return max(out_min, min(out_max, out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)))

def set_motor_position(motor, position):
    """Move yaw motor smoothly to a given position."""
    duty_cycle = map_range(position, -1, 1, MIN_DUTY, MAX_DUTY)
    motor.value = duty_cycle
    print(f"Yaw Motor Position: {position:.3f}, Duty Cycle: {duty_cycle:.5f}")

def pid_control(target_position, current_position, dt=0.1):
    """PID controller for smooth yaw tracking with speed limitation."""
    global previous_error, integral

    error = target_position - current_position
    integral += error * dt
    derivative = (error - previous_error) / dt
    previous_error = error

    output = Kp * error + Ki * integral + Kd * derivative

    # Limit the speed of change to prevent jerky motion
    output = max(-MAX_YAW_SPEED, min(MAX_YAW_SPEED, output))

    new_position = current_position + output
    return max(YAW_MIN, min(YAW_MAX, new_position))  # Keep within limits

def gen_frames():
    """Generate frames for the Flask app, adding object bounding boxes."""
    global yaw_position
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to get frame")
            break

        results = model(frame, imgsz=640)

        person_detected = False  # Reset flag for each frame

        # Process the detection results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if label == "person":
                    person_detected = True  # A person was detected
                    led.on()  # Turn on LED
                    target_yaw = (center_x - (FRAME_WIDTH / 2)) / (FRAME_WIDTH / 2)

                    # Smoothly update the yaw motor position
                    yaw_position = pid_control(target_yaw, yaw_position)
                    set_motor_position(yaw_motor, yaw_position)

                # Draw bounding boxes
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if not person_detected:
            led.off()  # Turn off LED if no person detected

        # Encode the frame as JPEG for web display
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Failed to encode frame")
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route to serve the video feed with bounding boxes."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run Flask in a separate thread to handle video feed and GPIO in parallel
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True))
    flask_thread.start()

    try:
        while True:
            # Keep the script running
            pass
    except KeyboardInterrupt:
        cap.release()
        flask_thread.join()
        print("Exiting...")
