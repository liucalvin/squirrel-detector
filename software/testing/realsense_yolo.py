import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO


def run_detection():
    model = YOLO('yolo11n.pt')

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    colorizer = rs.colorizer()

    # align depth to color stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)

            for result in results[0].boxes:
                class_id = int(result.cls)
                class_name = model.names[class_id]
                confidence = result.conf[0]
                
                # bounding box in color frame
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = f"{class_name}, {confidence:.2f}%"
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(color_image, (x1, y1 - 20), (x1 + 120, y1), (0, 0, 0), -1)
                cv2.putText(color_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                aligned_depth_frame = aligned_frames.get_depth_frame()

                if aligned_depth_frame:
                    # get depth measurement from center of depth frame
                    depth = aligned_depth_frame.get_distance(cx, cy)

                    if depth > 0:
                        # draw circle on the depth image
                        label = f"{class_name} {depth:.2f}m"
                        cv2.circle(colorized_depth, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.rectangle(colorized_depth, (cx, cy - 20), (cx + 120, cy), (0, 0, 0), -1)
                        cv2.putText(colorized_depth, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('RGB Image with Detections', color_image)
            cv2.imshow('Colorized Depth Image', colorized_depth)

            # exit loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


run_detection()
