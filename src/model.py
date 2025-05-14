import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
vehicle_class_ids = [2, 3, 5, 7]

def detect_vehicles(frame):
    resized = cv2.resize(frame, (640, 480))
    results = model(resized, verbose=False)[0]
    count = 0
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id in vehicle_class_ids:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[class_id]
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return resized, count
