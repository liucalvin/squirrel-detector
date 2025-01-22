import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

colorizer = rs.colorizer()

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)

        for result in results[0].boxes:
            class_id = int(result.cls)
            class_name = model.names[class_id]

            # bounding box in color frame
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # center of depth frame
            depth = depth_frame.get_distance(cx, cy)

            # draw bounding box on color frame
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(color_image, f"{class_name}, depth {depth:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow('RGB Image', color_image)
        cv2.imshow('Colorized Depth Image', colorized_depth)

        # exit loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
