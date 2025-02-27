import cv2
import time
import logging
from gpiozero import Servo
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Motor setup (Yaw for left/right, Pitch for up/down)
SERVO_YAW_PIN = 12
SERVO_PITCH_PIN = 13
yaw_servo = Servo(SERVO_YAW_PIN)
pitch_servo = Servo(SERVO_PITCH_PIN)

# Servo control limits
MAX_SPEED_DEG_PER_SEC = 30  # Max motor speed (30 degrees per second)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def move_servo(servo, target_pos, current_pos):
    """Move the servo gradually towards target_pos at a controlled speed."""
    step = 0.02  # Step delay
    max_step = MAX_SPEED_DEG_PER_SEC / 100  # Convert to fraction
    if target_pos > current_pos:
        while current_pos < target_pos:
            current_pos += min(max_step, target_pos - current_pos)
            servo.value = current_pos
            print(f"Moving servo to {current_pos:.2f}")
            time.sleep(step)
    else:
        while current_pos > target_pos:
            current_pos -= min(max_step, current_pos - target_pos)
            servo.value = current_pos
            print(f"Moving servo to {current_pos:.2f}")
            time.sleep(step)
    return current_pos

yaw_pos = 0  # Start at neutral position
pitch_pos = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        print("getting frame")
        if not ret:
            print("Failed to get frame")
            logging.error("Failed to get frame")
            break

        results = model(frame)
        print(f"getting results: {len(results)}")
        
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
                    # Normalize position to [-1, 1] range for `gpiozero`
                    target_yaw = (center_x - (FRAME_WIDTH / 2)) / (FRAME_WIDTH / 2)
                    target_pitch = (center_y - (FRAME_HEIGHT / 2)) / (FRAME_HEIGHT / 2)

                    # Move motors smoothly
                    yaw_pos = move_servo(yaw_servo, target_yaw, yaw_pos)
                    pitch_pos = move_servo(pitch_servo, target_pitch, pitch_pos)

except Exception as e:
    logging.error(f"Unexpected error: {e}", exc_info=True)
    print(f"Error: {e}")

finally:
    cap.release()
    print("Camera released, exiting...")

