from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Load and resize the image
image_path = r"Dynamic-Traffic-light-prediction-project/src/data/image.jpg"
image = Image.open(image_path).resize((450, 250))
image_arr = np.array(image)

# Convert RGB to BGR for OpenCV compatibility
image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)

# Load YOLOv8s model (pre-trained on COCO)
model = YOLO('yolov8x.pt')

# Run detection
results = model(image_bgr)[0]  # Get first result

# Class IDs for COCO: car=2, bus=5,truck=7, motorcycle=3 (bike is considered motorcycle)
car_count = bus_count = truck_count = bike_count = 0

# Draw bounding boxes
for box in results.boxes:
    cls_id = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    if cls_id == 2:  # car
        car_count += 1
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
    elif cls_id == 5:  # bus
        bus_count += 1
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    elif cls_id == 3:  # motorcycle
        bike_count += 1
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif cls_id == 7:  # Truck
        truck_count += 1
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display counts
print(f"{car_count} cars found")
print(f"{bus_count} buses found")
print(f"{bike_count} bikes found")
print(f"{truck_count} truck found")

# Convert back to RGB and show result
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
annotated_image = Image.fromarray(image_rgb)
annotated_image.show()
