from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Load YOLOv8 model
cap = cv2.VideoCapture(0)  # Open video stream (e.g., webcam)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

