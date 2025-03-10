import cv2
import numpy as np
from gpiozero import Servo
from time import sleep, time
import threading

# Constants
SERVO_PIN = 12
CAMERA_CHANNEL = 4
FRAME_WIDTH = 320  # Reduced resolution for better performance
FRAME_HEIGHT = 240
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
SERVO_ADJUSTMENT_FACTOR = 0.02  # Adjust this to control servo speed/sensitivity
SKIP_FRAMES = 2  # Process every nth frame to reduce load
DEAD_ZONE = 20  # Dead zone to prevent servo jitter

# Initialize servo
servo = Servo(SERVO_PIN)

# Load YOLO model (use yolov3-tiny for better performance)
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Handle different return types of getUnconnectedOutLayers()
if isinstance(output_layers_indices, np.ndarray):  # For newer OpenCV versions
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:  # For older OpenCV versions
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# Initialize camera
cap = cv2.VideoCapture(CAMERA_CHANNEL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Thread-safe variables
current_frame = None
frame_lock = threading.Lock()
stop_threads = False

def capture_frames():
    """Capture frames from the camera in a separate thread."""
    global current_frame, stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame
        sleep(0.01)  # Reduce CPU usage

def move_servo_to_center_object(center_x):
    """Move the servo to center the detected object."""
    error = center_x - CENTER_X
    if abs(error) > DEAD_ZONE:  # Dead zone to prevent jitter
        servo.value += (error / CENTER_X) * SERVO_ADJUSTMENT_FACTOR
        servo.value = max(-1, min(1, servo.value))  # Clamp servo value between -1 and 1

def detect_objects(frame):
    """Detect objects in the frame using YOLO."""
    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                # Object detected is a person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            center_x = x + w // 2
            center_y = y + h // 2

            # Move servo to center the object
            move_servo_to_center_object(center_x)

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

try:
    frame_count = 0
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Process every nth frame to reduce load
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        # Detect objects and move servo
        detect_objects(frame)

        # Sleep to reduce CPU usage
        sleep(0.01)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    stop_threads = True
    capture_thread.join()
    cap.release()
    servo.value = None  # Release the servo
