import cv2
import logging
import threading
import queue
import time
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Motor and GPIO setup
SERVO_YAW_PIN = 12
yaw_motor = PWMOutputDevice(SERVO_YAW_PIN, frequency=50)  # 50 Hz PWM

# Motor control limits
YAW_MIN = -1.0   # Leftmost position
YAW_MAX = 1.0    # Rightmost position
MIN_DUTY = 0.05
MAX_DUTY = 0.11

# PID Controller parameters
Kp = 0.1
Ki = 0.05
Kd = 0.02
MAX_YAW_SPEED = 0.05
previous_error = 0
integral = 0

# LED setup for person detection
led = LED(17)

# Video capture setup
FRAME_WIDTH = 320  # Reduced resolution
FRAME_HEIGHT = 240
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# YOLO model setup
model = YOLO("yolov5s.pt")  # Use smaller model for faster inference

# Shared variable for current yaw position
yaw_position = 0  # Start at the center

# Queue to hold frames for processing
frame_queue = queue.Queue(maxsize=1)

# Define the duty cycle range for PWM
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
    output = max(-MAX_YAW_SPEED, min(MAX_YAW_SPEED, output))
    new_position = current_position + output
    return max(YAW_MIN, min(YAW_MAX, new_position))  # Keep within limits

def capture_frames():
    """Capture frames from the camera in a separate thread."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()  # Discard old frame if the queue is full
        frame_queue.put(frame)
        time.sleep(0.033)  # Add delay to control FPS (around 30 FPS)

def process_frames():
    """Process frames from the queue and perform inference."""
    global yaw_position
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame)  # Inference

            person_detected = False  # Reset flag for each frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model.names[cls]

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if label == "person" and conf > 0.5:  # Confidence threshold for person detection
                        person_detected = True
                        led.on()  # Turn on LED when person detected

                        # Calculate target yaw position based on center_x
                        target_yaw = (center_x - (FRAME_WIDTH / 2)) / (FRAME_WIDTH / 2)

                        # Smoothly update the yaw motor position using PID
                        yaw_position = pid_control(target_yaw, yaw_position)
                        set_motor_position(yaw_motor, yaw_position)

            # If no person was detected, turn the LED off
            if not person_detected:
                led.off()

            # Show the processed frame with bounding boxes
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Show the frame with bounding boxes
            cv2.imshow("Frame", frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

try:
    # Start capture and processing threads
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_frames)

    capture_thread.start()
    process_thread.start()

    # Flask app to display the video feed (optional)
    from flask import Flask, Response
    app = Flask(__name__)

    @app.route('/')
    def index():
        return "<h1>Live Stream</h1>"

    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=5000, threaded=True)

except Exception as e:
    logging.error(f"Unexpected error: {e}", exc_info=True)
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released, exiting...")
