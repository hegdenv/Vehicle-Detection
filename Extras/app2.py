import streamlit as st
import cv2
import torch
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import tempfile
import os
from collections import defaultdict
import time

class VehicleDetectionSystem:
    def __init__(self, confidence_threshold=0.25):
        # Initialize YOLO model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(self.device)
        self.model.conf = confidence_threshold
        self.model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # Initialize tracking and counting variables
        self.vehicle_tracker = {}
        self.counted_vehicles = set()
        self.next_vehicle_id = 1
        self.trajectories = defaultdict(list)
        self.counting_line_y = None

        # Tracking parameters
        self.max_disappeared = 15
        self.min_frames_tracked = 2

    def detect_vehicles(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = self.model(rgb_frame)
        
        detections = []
        for det in results.xyxy[0].cpu().numpy():
            if det[5] in [2, 3, 5, 7]:
                conf = det[4]
                x1, y1, x2, y2 = map(int, det[:4])
                w = x2 - x1
                h = y2 - y1
                if w > 20 and h > 20:
                    detections.append({
                        'bbox': (x1, y1, w, h),
                        'center': (x1 + w//2, y1 + h//2),
                        'confidence': conf
                    })
        return detections

    def update_tracks(self, detections):
        matched_tracks = {}
        unmatched_detections = []

        for vehicle_id in list(self.vehicle_tracker.keys()):
            self.vehicle_tracker[vehicle_id]['disappeared'] += 1

        for detection in detections:
            matched = False
            detection_center = detection['center']

            for vehicle_id, track_info in self.vehicle_tracker.items():
                track_center = track_info['center']
                distance = np.sqrt(
                    (detection_center[0] - track_center[0])**2 +
                    (detection_center[1] - track_center[1])**2
                )
                if distance < 50:
                    matched_tracks[vehicle_id] = {
                        'bbox': detection['bbox'],
                        'center': detection_center,
                        'disappeared': 0
                    }
                    self.trajectories[vehicle_id].append(detection_center)
                    matched = True
                    break

            if not matched:
                unmatched_detections.append(detection)

        for detection in unmatched_detections:
            self.vehicle_tracker[self.next_vehicle_id] = {
                'bbox': detection['bbox'],
                'center': detection['center'],
                'disappeared': 0
            }
            self.trajectories[self.next_vehicle_id].append(detection['center'])
            self.next_vehicle_id += 1

        self.vehicle_tracker.update(matched_tracks)

        for vehicle_id in list(self.vehicle_tracker.keys()):
            if self.vehicle_tracker[vehicle_id]['disappeared'] > self.max_disappeared:
                del self.vehicle_tracker[vehicle_id]
                del self.trajectories[vehicle_id]

    def count_vehicles(self, frame_height):
        if self.counting_line_y is None:
            self.counting_line_y = int(frame_height * 0.6)

        for vehicle_id, track_info in self.vehicle_tracker.items():
            if vehicle_id not in self.counted_vehicles:
                trajectory = self.trajectories[vehicle_id]
                if len(trajectory) >= self.min_frames_tracked:
                    prev_points = trajectory[-2:]
                    if len(prev_points) >= 2:
                        prev_y = prev_points[0][1]
                        curr_y = prev_points[1][1]
                        if (prev_y < self.counting_line_y and curr_y >= self.counting_line_y) or \
                           (prev_y > self.counting_line_y and curr_y <= self.counting_line_y):
                            self.counted_vehicles.add(vehicle_id)

    def draw_visualization(self, frame):
        cv2.line(frame, (0, self.counting_line_y),
                (frame.shape[1], self.counting_line_y), (0, 255, 255), 2)

        for vehicle_id, track_info in self.vehicle_tracker.items():
            x, y, w, h = track_info['bbox']
            color = (0, 255, 0) if vehicle_id in self.counted_vehicles else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID: {vehicle_id}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            trajectory = self.trajectories[vehicle_id]
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 2)

        cv2.putText(frame, f"Vehicle Count: {len(self.counted_vehicles)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def process_frame(self, frame):
        detections = self.detect_vehicles(frame)
        self.update_tracks(detections)
        self.count_vehicles(frame.shape[0])
        return self.draw_visualization(frame)

def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

    
def main():
    # st.title("Vehicle Analysis Application")
    
    # # Sidebar for choosing functionality
    # analysis_type = st.sidebar.selectbox(
    #     "Choose Analysis Type",
    #     ["Vehicle Classification", "Vehicle Counting in Video"]
    # )
    
    # if analysis_type == "Vehicle Classification":
    #     st.header("Vehicle Image Classification")
        
    #     # Load the classification model
    #     try:
    #         model_path = st.text_input("Enter path to classification model (.keras file):")
    #         if model_path and os.path.exists(model_path):
    #             classification_model = tf.keras.models.load_model(model_path)
    #             st.success("Model loaded successfully!")
            
    #             # File uploader for images
    #             uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
                
    #             if uploaded_file is not None:
    #                 # Convert uploaded file to image
    #                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #                 img = cv2.imdecode(file_bytes, 1)
                    
    #                 # Display the uploaded image
    #                 st.image(img, caption='Uploaded Image', channels="BGR")
                    
    #                 if st.button('Classify Image'):
    #                     # Preprocess and classify
    #                     processed_img = preprocess_image(img)
    #                     prediction = classification_model.predict(processed_img)
    #                     result = "Vehicle" if prediction[0][0] >= 0.5 else "Non-Vehicle"
    #                     st.success(f"Classification Result: {result}")
            
    #     except Exception as e:
    #         st.error(f"Error loading model: {str(e)}")
    st.title("Vehicle Analysis Application")
    
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Vehicle Classification", "Vehicle Counting in Video"]
    )
    
    if analysis_type == "Vehicle Classification":
        st.header("Vehicle Image Classification")
        
        # Load the model at startup
        classification_model = load_classification_model()
        
        if classification_model:
            # File uploader for images
            uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                # Display the uploaded image
                st.image(img, caption='Uploaded Image', channels="BGR")
                
                if st.button('Classify Image'):
                    # Add a spinner while processing
                    with st.spinner('Classifying image...'):
                        # Preprocess and classify
                        processed_img = preprocess_image(img)
                        prediction = classification_model.predict(processed_img)
                        result = "Vehicle" if prediction[0][0] >= 0.5 else "Non-Vehicle"
                        
                        # Display result with styling
                        st.markdown(f"""
                        <div style='padding: 20px; border-radius: 10px; background-color: #e8f4ea;'>
                            <h3 style='text-align: center; color: #28a745;'>Classification Result</h3>
                            <h2 style='text-align: center; color: #28a745;'>{result}</h2>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:  # Vehicle Counting
        st.header("Vehicle Counting in Video")
        
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # Video processing
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("Error: Could not open video file")
            else:
                # Initialize detector
                detector = VehicleDetectionSystem(confidence_threshold=0.25)
                
                # Create placeholders for video frame and count
                frame_placeholder = st.empty()
                count_placeholder = st.empty()
                stop_button = st.button("Stop Processing")
                
                # Display initial count
                count_placeholder.markdown(f"<h2 style='text-align: center;'>Vehicle Count: 0</h2>", unsafe_allow_html=True)
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame = detector.process_frame(frame)
                    
                    # Convert BGR to RGB for display
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update the frame
                    frame_placeholder.image(processed_frame_rgb)
                    
                    # Update the count in a single, centered display
                    count_placeholder.markdown(
                        f"<h2 style='text-align: center;'>Vehicle Count: {len(detector.counted_vehicles)}</h2>",
                        unsafe_allow_html=True
                    )
                    
                    # Add small delay
                    time.sleep(0.1)
                
                cap.release()
                os.unlink(tfile.name)
                
                st.success("Video processing complete!")

if __name__ == "__main__":
    main()