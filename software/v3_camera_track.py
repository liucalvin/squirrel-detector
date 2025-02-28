import cv2
import time
import logging
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Motor setup
yaw_motor = PWMOutputDevice(12)  # Left/Right movement
pitch_motor = PWMOutputDevice(13)  # Up/Down movement (DISABLED)

# Define yaw motor limits
YAW_MIN = -1  # Left limit
YAW_MAX = 1   # Right limit

# Pitch is disabled for now
FIXED_PITCH = 1.0  # Set a fixed pitch position

# Define the duty cycle range
MIN_DUTY = 0.05
MAX_DUTY = 0.11

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
RESOLUTION=736

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# LED indicator
led = LED(17)

def map_range(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another while keeping it within limits."""
    return max(out_min, min(out_max, out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)))

def set_motor_position(motor, position):
    """Move motor smoothly to a given position."""
    duty_cycle = map_range(position, -1, 1, MIN_DUTY, MAX_DUTY)
    motor.value = duty_cycle
    print(f"Motor on GPIO {motor.pin.number}: Position {position:.3f}, Duty {duty_cycle:.5f}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            logging.error("Failed to get frame")
            break

        results = model(frame, imgsz=RESOLUTION)

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

                    # Map detected position to yaw motor
                    target_yaw = map_range(center_x, 0, FRAME_WIDTH, YAW_MIN, YAW_MAX)

                    # Move only the yaw motor
                    set_motor_position(yaw_motor, target_yaw)

                    print(f"Person detected at X={center_x:.0f} → Yaw: {target_yaw:.2f}")

        # Turn off LED if no person is detected
        if not person_detected:
            led.off()
            print("No person detected. LED OFF")

except Exception as e:
    logging.error(f"Unexpected error: {e}", exc_info=True)
    print(f"Error: {e}")

finally:
    cap.release()
    print("Camera released, exiting...")
