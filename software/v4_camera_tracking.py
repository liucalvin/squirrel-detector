import cv2
import logging
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

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
model = YOLO("yolov8n.pt")
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


try:
    while cap.isOpened():
        ret, frame = cap.read()
        print("getting frame")
        if not ret:
            print("Failed to get frame")
            logging.error("Failed to get frame")
            break

        results = model(frame, imgsz=640)
        print(f"getting results: {len(results)}")

        person_detected = False  # Reset flag for each frame

        for result in results:
            print(f"getting box: {len(result.boxes)}")
            for box in result.boxes:
                print("calculating box")
                print(f"{box.xyxy[0]} {box.conf[0].item()} {box.cls[0].item()}")

                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                print(f"Detected {label} at ({center_x:.0f}, {center_y:.0f}) with confidence {conf:.2f}")

                if label == "person":
                    person_detected = True  # A person was detected
                    led.on()
                    print("Person detected! LED ON")

                    # Calculate target yaw position (-1 to 1 based on center_x position)
                    target_yaw = (center_x - (FRAME_WIDTH / 2)) / (FRAME_WIDTH / 2)

                    # Smoothly update the yaw motor position
                    yaw_position = pid_control(target_yaw, yaw_position)
                    set_motor_position(yaw_motor, yaw_position)

        # If no person was detected in this frame, turn the LED off
        if not person_detected:
            led.off()
            print("No person detected. LED OFF")

except Exception as e:
    logging.error(f"Unexpected error: {e}", exc_info=True)
    print(f"Error: {e}")

finally:
    cap.release()
    print("Camera released, exiting...")
