import cv2
import time
import traceback
from gpiozero import Servo
from ultralytics import YOLO

# GPIO pin assignments
SERVO_YAW_PIN = 12   # Left/Right movement
SERVO_PITCH_PIN = 13  # Up/Down movement

# Initialize servos
try:
    yaw_servo = Servo(SERVO_YAW_PIN)
    pitch_servo = Servo(SERVO_PITCH_PIN)
    print("Servos initialized successfully.")
except Exception as e:
    print(f"Servo initialization failed: {e}")
    with open("error.log", "w") as f:
        f.write(traceback.format_exc())
    exit(1)

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"YOLO initialization failed: {e}")
    with open("error.log", "w") as f:
        f.write(traceback.format_exc())
    exit(1)


