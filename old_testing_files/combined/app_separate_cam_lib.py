from flask import Flask, render_template, Response
import cv2
import torch
import pyrealsense2 as rs
import numpy as np
from gpiozero import Servo
from time import sleep
from ultralytics import YOLO  # Import YOLO from Ultralytics

app = Flask(__name__)

# Servo setup
servo_pin = 12  # GPIO pin connected to the servo
servo = Servo(servo_pin)

# Initialize the current servo position
current_position = 0  # Neutral position (0)

# Define the step size for small movements
step_size = 0.2  # Adjust this value to control the movement amount

# Define a deadband to filter small changes
deadband = 0.05  # Ignore changes smaller than this value

# Camera setup for RGB
camera = cv2.VideoCapture(4, cv2.CAP_V4L2)  # Use 4 for the RGB camera

# Load YOLOv11n model using Ultralytics
model = YOLO("yolo11n.pt")  # Load YOLOv11n model

# Load class names from coco.names
with open("coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Configure RealSense pipeline for depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start RealSense streaming
pipeline.start(config)

EXIT_PRESSURE = 100  # psi
NOZZLE_HEIGHT = 0.2  # meters above ground

def generate_rgb_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model(frame)  # Run inference
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

                # Draw bounding boxes on the frame
                for box, conf, cls in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{class_names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_depth_frames():
    try:
        while True:
            # Wait for a coherent pair of frames: depth
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # Convert depth frame to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())

            # Get the center depth value
            height, width = depth_image.shape
            center_x = width // 2
            center_y = height // 2
            center_depth = depth_image[center_y, center_x] / 1000.0  # Depth in meters

            # Apply colormap for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Overlay the center depth value on the image
            size = 4
            cv2.circle(depth_colormap, (center_x, center_y), radius=size + 2, color=(0, 0, 0), thickness=-1)
            cv2.line(depth_colormap, (center_x - size, center_y - size), (center_x + size, center_y + size), (255, 255, 255), 2)
            cv2.line(depth_colormap, (center_x + size, center_y - size), (center_x - size, center_y + size), (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Center Depth: {round(center_depth, 1)} m", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', depth_colormap)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Stop streaming
        pipeline.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(generate_rgb_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(generate_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/move_left')
def move_left():
    global current_position
    new_position = min(1, current_position + step_size)  # Adjusted for flipped motor
    if abs(new_position - current_position) >= deadband:
        current_position = new_position
        servo.value = current_position
        sleep(0.1)  # Add a small delay
    return f"Moved Left to {current_position}"

@app.route('/move_right')
def move_right():
    global current_position
    new_position = max(-1, current_position - step_size)  # Adjusted for flipped motor
    if abs(new_position - current_position) >= deadband:
        current_position = new_position
        servo.value = current_position
        sleep(0.1)  # Add a small delay
    return f"Moved Right to {current_position}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
