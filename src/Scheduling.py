import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Vehicle class IDs from COCO dataset: car, motorcycle, bus, truck
vehicle_class_ids = [2, 3, 5, 7]

# List of video paths (each one is a lane/intersection)
video_list = [
    r"Dynamic-Traffic-light-prediction-project/src/data/A.mp4",
    r"Dynamic-Traffic-light-prediction-project/src/data/B.mp4",
    r"Dynamic-Traffic-light-prediction-project/src/data/C.mp4",
    r"Dynamic-Traffic-light-prediction-project/src/data/D.mp4"
]

# Open video captures
caps = [cv2.VideoCapture(video) for video in video_list]

# Timing settings
base_time = 10  # Minimum green time
max_time = 180  # Maximum green time


def count_vehicles_with_yolo(frame):
    resized_frame = cv2.resize(frame, (640, 480))
    results = model(resized_frame, verbose=False)[0]
    count = 0

    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id in vehicle_class_ids:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[class_id]
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return count, resized_frame


while True:
    vehicle_counts = []
    frames = []

    # Step 1: Get vehicle counts from one frame of each video
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            vehicle_counts.append(0)
            frames.append(None)
            continue

        count, processed = count_vehicles_with_yolo(frame)
        vehicle_counts.append(count)
        frames.append(processed)

    total_vehicles = sum(vehicle_counts)

    if total_vehicles == 0:
        print("No vehicles detected in any lane. All signals stay YELLOW.")
        time.sleep(5)
        continue

    # Step 2: Dynamically assign green times based on vehicle volume
    green_times = [
        int(base_time + ((count / total_vehicles) * (max_time - base_time)))
        if total_vehicles > 0 else base_time
        for count in vehicle_counts
    ]

    # Step 3: Loop through each lane and simulate signal
    i=0
    while (i<4):
        print(f"\n===== Lane {i+1} =====")
        print(f"Vehicles Detected: ",vehicle_counts[i])
        print(f"Green Light Time: {green_times[i]} seconds")

        print("RED Lights for other lanes...")
        time.sleep(1)

        print(f"GREEN Light ON for Lane {i+1}")
        
        if frames[i] is not None:
          cv2.imshow(f"Lane {i+1}", frames[i])
          cv2.waitKey(1)  

          start = time.time()
          while time.time() - start < green_times[i]:
            if cv2.waitKey(100) & 0xFF == ord('q'):
              break

        print("YELLOW Light ON")
        time.sleep(3)
        i+=1
        # Close preview window for previous lane
        cv2.destroyWindow(f"Lane {i}")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Cleanup
    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()
