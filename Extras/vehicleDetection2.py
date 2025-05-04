import cv2
import torch
import numpy as np  # Ensure this is imported for numpy operations
import sys
import os

# Add the 'sort' folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))

from sort import Sort  # Assuming the sort module is set up correctly

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize video capture
video_path = 'Datasets/carv.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize SORT tracker
tracker = Sort()

# Define YOLO classes for vehicles (0 for 'person', 2 for 'car', etc.)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

vehicle_count = 0
frame_skip_rate = 5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for speed
    frame_count += 1
    if frame_count % frame_skip_rate != 0:
        continue

    # Perform detection with YOLOv5
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Filter out only vehicle detections with high confidence
    filtered_boxes = []
    for _, row in detections.iterrows():
        if row['class'] in vehicle_classes and row['confidence'] > 0.5:
            filtered_boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence']])

    # Convert to numpy array for SORT tracker
    filtered_boxes = np.array(filtered_boxes)

    # Check if filtered_boxes is not empty and has the correct shape
    if filtered_boxes.size > 0 and filtered_boxes.shape[1] == 5:
        # Keep only the x_min, y_min, x_max, y_max values
        filtered_boxes = filtered_boxes[:, :4]
    else:
        filtered_boxes = np.empty((0, 4))  # If no vehicles are detected, set filtered_boxes to an empty array

    # Update tracker
    tracked_objects = tracker.update(filtered_boxes)

    # Draw bounding boxes and count vehicles
    for x1, y1, x2, y2, obj_id in tracked_objects:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Update vehicle count with unique IDs
    vehicle_count += len(np.unique([obj_id for _, _, _, _, obj_id in tracked_objects]))

    # Display total vehicle count on the frame
    cv2.putText(frame, f'Total Vehicles: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Vehicle Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
