import cv2
import time
import logging
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Yaw Motor Setup (Left/Right)
yaw_motor = PWMOutputDevice(12)  # GPIO for yaw movement

# Define yaw motor limits
YAW_MIN = -1   # Leftmost position
YAW_MAX = 1    # Rightmost position

# Define the duty cycle range for PWM
MIN_DUTY = 0.05
MAX_DUTY = 0.11

# Camera Frame Size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CENTER_X = FRAME_WIDTH / 2  # Target center for tracking

# PID Controller Parameters (Tuned for smoother movement)
Kp = 0.3   # Proportional gain (controls how much it moves based on error)
Ki = 0.05  # Integral gain (helps with steady-state error)
Kd = 0.02  # Derivative gain (reduces overshoot)

MAX_YAW_SPEED = 0.05  # Limits how fast the motor can move per update
previous_error = 0
integral = 0

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# LED Indicator
led = LED(17)

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
    if output > MAX_YAW_SPEED:
        output = MAX_YAW_SPEED
    elif output < -MAX_YAW_SPEED:
        output = -MAX_YAW_SPEED

    new_position = current_position + output
    return max(YAW_MIN, min(YAW_MAX, new_position))  # Keep within limits

try:
    yaw_position = 0  # Start at center

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            logging.error("Failed to get frame")
            break

        results = model(frame, imgsz=848)

        person_detected = False  # Reset flag for each frame

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                if label == "person":
                    person_detected = True  # A person is detected
                    led.on()

                    # Calculate the center of the bounding box
                    center_x = (x1 + x2) / 2

                    # Compute how far the detected person is from the center
                    error_x = (center_x - CENTER_X) / (FRAME_WIDTH / 2)  # Normalize to [-1, 1]

                    # Smoothly adjust yaw to keep the person centered
                    yaw_position = pid_control(-error_x, yaw_position)  # Negative sign to correct direction
                    set_motor_position(yaw_motor, yaw_position)

                    print(f"Person detected at X={center_x:.0f} → Error: {error_x:.2f}, Smoothed Yaw: {yaw_position:.2f}")

        # Turn off LED if no person is detected
        if not person_detected:
            led.off()
            print("No person detected. LED OFF")

        time.sleep(0.1)  # Small delay for PID update

except Exception as e:
    logging.error(f"Unexpected error: {e}", exc_info=True)
    print(f"Error: {e}")

finally:
    cap.release()
    print("Camera released, exiting...")
