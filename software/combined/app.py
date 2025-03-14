from flask import Flask, render_template, Response, request
import pyrealsense2 as rs
import numpy as np
import cv2
import serial
from time import sleep
from ultralytics import YOLO
from motor_controller import MotorController

app = Flask(__name__)

# Create an instance of the MotorWaterController
controller = MotorController()

# Load YOLOv11n model using Ultralytics
model = YOLO("yolo11n.pt")  # Ensure the model file is in the same directory

# Load class names from coco.names
with open("coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Configure RealSense pipeline for both RGB and depth
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream (640x480)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    # Depth stream (640x480)

# Start RealSense streaming
try:
    pipeline.start(config)
except RuntimeError as e:
    print(f"Failed to start RealSense pipeline: {e}")
    exit(1)

EXIT_PRESSURE = 100  # psi
NOZZLE_HEIGHT = 0.2  # meters above ground

def generate_rgb_frames():
    try:
        while True:
            # Wait for a coherent pair of frames: RGB and depth
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RGB frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Perform object detection
            results = model(color_image)  # Run inference
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

                # Draw bounding boxes on the frame
                for box, conf, cls in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{class_names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', color_image)
            if not ret:
                print("Failed to encode RGB frame.")
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Stop streaming
        pipeline.stop()

def generate_depth_frames():
    try:
        while True:
            # Wait for a coherent pair of frames: RGB and depth
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

# Motor and water control routes
@app.route('/move_left')
def move_left():
    return controller.move_left()

@app.route('/move_right')
def move_right():
    return controller.move_right()

@app.route('/move_up')
def move_up():
    return controller.move_up()

@app.route('/move_down')
def move_down():
    return controller.move_down()

@app.route('/water_on')
def water_on():
    return controller.turn_water_on()

@app.route('/water_off')
def water_off():
    return controller.turn_water_off()
  
@app.route('/water_toggle')
def water_toggle():
    if controller.water_on == 0:
        return controller.turn_water_on()
    else:
        return controller.turn_water_off()

# Custom serial command route
@app.route('/send_command', methods=['POST'])
def send_command():
    command = request.form.get('command')  # Get the command from the form
    if command:
        controller.send_custom_command(command)
        return f"Sent custom command: {command}"
    return "No command provided."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
