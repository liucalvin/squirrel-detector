import cv2
import time
import logging
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Yaw Motor Setup
SERVO_YAW_PIN = 12
yaw_motor = PWMOutputDevice(SERVO_YAW_PIN, frequency=50)  # 50 Hz PWM

# Yaw motor limits
YAW_MIN = -1.0   # Leftmost position
YAW_MAX = 1.0    # Rightmost position

# Define the duty cycle range for PWM (servo-specific)
MIN_DUTY = 0.05
MAX_DUTY = 0.11

# LED setup (turns on when a person is detected)
led = LED(17)

# Video capture setup
FRAME_WIDTH = 720
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


def map_range(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another while keeping it within limits."""
    return max(out_min, min(out_max, out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)))


def set_motor_position(motor, position):
    """Move yaw motor smoothly to a given position."""
    duty_cycle = map_range(position, -1, 1, MIN_DUTY, MAX_DUTY)
    motor.value = duty_cycle
    print(f"Yaw Motor Position: {position:.3f}, Duty Cycle: {duty_cycle:.5f}")


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            logging.error("Failed to get frame")
            break

        results = model(frame, imgsz=640)

        person_detected = False  # Reset flag for each frame

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                center_x = (x1 + x2) / 2

                if label == "person":
                    person_detected = True  # A person was detected
                    led.on()

                    # Calculate target yaw position (-1 to 1 based on center_x position)
                    target_yaw = (center_x - CENTER_X) / CENTER_X

                    # Move the yaw motor directly
                    set_motor_position(yaw_motor, target_yaw)

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
