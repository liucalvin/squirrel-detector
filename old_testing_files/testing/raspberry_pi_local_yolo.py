from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')  # Load YOLOv8 model
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Open video stream (e.g., webcam)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break
  
    time.sleep(1)

    # frame = cv2.resize(frame, (320, 320))
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0] # bounding box
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"Detected {label} at ({center_x:.0f}, {center_y:.0f}) with confidence {conf:.2f}")

    time.sleep(2)
    # annotated_frame = results[0].plot()
    
    # cv2.imshow('YOLO Detection', annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()

