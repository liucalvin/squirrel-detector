import cv2
import time
import logging
from gpiozero import PWMOutputDevice, LED
from ultralytics import YOLO

# Setup logging
logging.basicConfig(filename="error.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Motor setup (Yaw for left/right, Pitch for up/down)
SERVO_YAW_PIN = 12
SERVO_PITCH_PIN = 13

yaw_servo = PWMOutputDevice(SERVO_YAW_PIN, frequency=50)  # 50 Hz PWM
pitch_servo = PWMOutputDevice(SERVO_PITCH_PIN, frequency=50)

# Servo control limits
SERVO_MIN = 900
SERVO_MAX = 1500
SERVO_RANGE = 2048

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def map_to_servo_range(target, min_value, max_value):
    """Maps a target position from one range to another."""
    return max(min_value, min(max_value, int(target)))

def move_servo(servo, target_pos):
    """Move the servo to a specific position within limits."""
    mapped_value = map_to_servo_range(target_pos, SERVO_MIN, SERVO_MAX)
    if False:  # Prevent actual movement
        servo.value = mapped_value / SERVO_RANGE  # Normalize to 0-1
    print(f"Moving servo to {target_pos} (Mapped: {mapped_value})")

# Start servos in the center
yaw_pos = (SERVO_MIN + SERVO_MAX) / 2  
pitch_pos = (SERVO_MIN + SERVO_MAX) / 2

led = LED(17)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        print("getting frame")
        if not ret:
            print("Failed to get frame")
            logging.error("Failed to get frame")
            break

        results = model(frame, imgsz=1280)
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

                    # Map center position to servo range
                    target_yaw = SERVO_MIN + (center_x / FRAME_WIDTH) * (SERVO_MAX - SERVO_MIN)
                    target_pitch = SERVO_MIN + (center_y / FRAME_HEIGHT) * (SERVO_MAX - SERVO_MIN)

                    led.on()
                    print("Person detected! LED ON")

                    if False:  # Prevent actual motor movement
                        move_servo(yaw_servo, target_yaw)   # Move camera left/right
                        move_servo(pitch_servo, target_pitch)  # Move nozzle up/down

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
