import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from fluid_velocity_calculator import find_launch_angle, exit_velocity

class CameraController:
    def __init__(self, motor_controller, selective_tracking=True, motor_tracking=False):
        # Load YOLOv11n model using Ultralytics
        self.model = YOLO("yolo11n.pt")  # Ensure the model file is in the same directory

        # Load class names from coco.names
        with open("coco.names", "r") as f:
            self.class_names = f.read().strip().split("\n")

        # Configure RealSense pipeline for both RGB and depth
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(stream_type=rs.stream.color, width=640, height=480, format=rs.format.bgr8, framerate=15)  # RGB stream (640x480)
        config.enable_stream(stream_type=rs.stream.depth, width=640, height=480, format=rs.format.z16, framerate=15)    # Depth stream (640x480)

        # Start RealSense streaming
        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            print(f"Failed to start RealSense pipeline: {e}")
            exit(1)

        # Motor controller instance
        self.motor_controller = motor_controller

        # Tracking settings
        self.selective_tracking = selective_tracking  # Toggle selective tracking
        self.motor_tracking = motor_tracking  # Toggle motor tracking
        self.tracked_classes = [0, 21]  # Class IDs for "person" and "bear"

        # Proportional control gain
        self.kp = 0.08  # Proportional gain (adjust this value for responsiveness)

        # Store the latest detection results
        self.latest_boxes = []  # Stores bounding boxes for detected objects
        self.latest_class_ids = []  # Stores class IDs for detected objects
        self.latest_confidences = []  # Stores confidence scores for detected objects
        self.latest_depths = []  # Stores depth values for detected objects

        # Water control state
        self.water_on = False  # Track whether water is currently on

    def generate_rgb_frames(self):
        try:
            while True:
                # Wait for a coherent pair of frames: RGB and depth
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert RGB frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())

                # Perform object detection
                results = self.model(color_image, imgsz=320)  # Run inference

                # Clear previous detection results
                self.latest_boxes = []
                self.latest_class_ids = []
                self.latest_confidences = []
                self.latest_depths = []

                # Filter results for selective tracking
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

                    # Store bounding boxes, class IDs, and confidences for depth processing
                    for box, conf, cls in zip(boxes, confidences, class_ids):
                        if not self.selective_tracking or int(cls) in self.tracked_classes:
                            if conf > 0.5:  # Only track if confidence > 50%
                                self.latest_boxes.append(box)
                                self.latest_class_ids.append(int(cls))
                                self.latest_confidences.append(conf)

                                # Draw bounding boxes and labels only for tracked classes
                                x1, y1, x2, y2 = map(int, box)
                                label = f"{self.class_names[int(cls)]} {conf:.2f}"
                                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # Motor tracking logic
                                if self.motor_tracking and int(cls) in self.tracked_classes:
                                    self._track_object(box, color_image.shape[1])

                # Toggle water based on detected objects
                if self.latest_boxes:  # If objects are detected
                    if not self.water_on:  # If water is off, turn it on
                        self.motor_controller.turn_water_on()
                        self.water_on = True
                else:  # If no objects are detected
                    if self.water_on:  # If water is on, turn it off
                        self.motor_controller.turn_water_off()
                        self.water_on = False

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
            self.pipeline.stop()

    def generate_depth_frames(self):
        try:
            while True:
                # Wait for a coherent pair of frames: RGB and depth
                frames = self.pipeline.wait_for_frames()
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

                # Draw depth values for detected objects
                self.latest_depths = []  # Reset depth values
                for box, cls, conf in zip(self.latest_boxes, self.latest_class_ids, self.latest_confidences):
                    if not self.selective_tracking or cls in self.tracked_classes:
                        if conf > 0.5:  # Only process if confidence > 50%
                            x1, y1, x2, y2 = map(int, box)
                            center_x_box = (x1 + x2) // 2
                            center_y_box = (y1 + y2) // 2
                            depth_value = depth_image[center_y_box, center_x_box] / 1000.0  # Depth in meters
                            self.latest_depths.append(depth_value)

                            # Draw a dot at the center of the bounding box
                            cv2.circle(depth_colormap, (center_x_box, center_y_box), radius=5, color=(0, 255, 0), thickness=-1)
                            cv2.putText(depth_colormap, f"{round(depth_value, 1)} m", (center_x_box + 10, center_y_box),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Find the object closest to the center and set motor angle
                if self.motor_tracking and self.latest_depths:
                    closest_depth = min(self.latest_depths)  # Find the closest object
                    launch_angle = find_launch_angle(exit_velocity(100), 0.3, closest_depth)
                    if launch_angle is not None:
                        self.motor_controller.move_up(step_size=int(launch_angle - self.motor_controller.motor_angle_2))

                # Encode the frame to JPEG
                ret, buffer = cv2.imencode('.jpg', depth_colormap)
                frame = buffer.tobytes()

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            # Stop streaming
            self.pipeline.stop()

    def _track_object(self, box, frame_width):
        """
        Track the object using proportional control to adjust motor step size.
        """
        x1, y1, x2, y2 = box
        center_x_box = (x1 + x2) // 2  # Center of the bounding box
        frame_center = frame_width // 2  # Center of the frame

        # Calculate the error (difference between object center and frame center)
        error = center_x_box - frame_center

        # Define a dead zone to avoid jitter
        dead_zone = 20  # Pixels
        if abs(error) < dead_zone:
            return  # Object is close enough to the center, no movement needed

        # Calculate step size using proportional control
        step_size = int(self.kp * abs(error))

        # Limit step size to a reasonable range (e.g., 1 to 20 degrees)
        step_size = max(1, min(step_size, 20))

        # Move motors based on error direction
        if error < 0:
            self.motor_controller.move_left(step_size)  # Move left if object is to the left
        else:
            self.motor_controller.move_right(step_size)  # Move right if object is to the right

    def find_launch_angle(self, v0, h0, z_target):
        """
        Find the shallower launch angle for the stream to hit the target depth.
        """
        g = 9.81  # Gravity (m/s^2)
        
        # Function to calculate time to reach the target depth (z_target)
        def time_to_hit_target(theta):
            """Solve for time given the launch angle."""
            theta_rad = np.radians(theta)
            v0y = v0 * np.sin(theta_rad)
            discriminant = v0y**2 + 2 * g * (h0 - z_target)
            if discriminant < 0:
                return None  # No valid time
            t = (v0y - np.sqrt(discriminant)) / g  # Use negative root for shallower trajectory
            return t

        # Function to calculate horizontal distance at a given time
        def horizontal_distance(theta, t):
            """Calculate horizontal distance given time and angle."""
            return v0 * np.cos(np.radians(theta)) * t

        # Try different angles and find the shallower one
        angles = np.linspace(1, 89, 1000)  # Sweep from 1 to 89 degrees
        best_angle = None
        min_diff = float('inf')

        for angle in angles:
            t = time_to_hit_target(angle)
            if t is None:
                continue  # Skip invalid times
            
            x = horizontal_distance(angle, t)
            diff = abs(x - z_target)  # Difference between actual and target horizontal distance
            
            if diff < min_diff:
                min_diff = diff
                best_angle = angle
        
        return best_angle

    def set_motor_tracking(self, enabled):
        """
        Enable or disable motor tracking.
        """
        self.motor_tracking = enabled
