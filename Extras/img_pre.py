import streamlit as st
import cv2
import torch
import numpy as np
import tensorflow as tf
import keras.utils as image
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
                            self .counted_vehicles.add(vehicle_id)

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

def preprocess_image(file_like):
    img = image.load_img(file_like, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

def classify_image(file_like, model):
    try:
        img = preprocess_image(file_like)
        prediction = model.predict(img)
        confidence = float(prediction[0][0]) * 100  # Convert to percentage
        
        if confidence >= 70:  # 70% threshold
            return "Vehicle", confidence
        else:
            return "Non-Vehicle", confidence
            
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return "Error", 0.0

def load_classification_model():
    MODEL_PATH = r"C:\Users\hp\Desktop\Visual Computing\Models\mymodel.keras"
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def handle_video_upload(uploaded_file):
    if uploaded_file is not None:
        # Check if the uploaded file is different from the current one
        if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
            # Clean up existing video resources
            if 'cap' in st.session_state and st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            
            if 'temp_file_path' in st.session_state and st.session_state.temp_file_path is not None:
                try:
                    os.unlink(st.session_state.temp_file_path)
                except:
                    pass
            
            # Save new video file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.session_state.temp_file_path = tfile.name
            tfile.close()
            
            # Update session state
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.processing = False
            st.session_state.vehicle_count = 0
            
            # Reset video capture
            if 'cap' in st.session_state and st.session_state.cap is not None:
                st.session_state.cap.release()
            st.session_state.cap = None
            
            return True
    return False

def main():
    st.title("Vehicle Analysis Application")

    # Initialize session state variables
    if 'vehicle_count' not in st.session_state:
        st.session_state.vehicle_count = 0
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'temp_file_path' not in st.session_state:
        st.session_state.temp_file_path = None
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    
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
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                # Display the uploaded image
                st.image(temp_file_path, caption='Uploaded Image', use_column_width=True)
                
                if st.button('Classify Image'):
                    with st.spinner('Classifying image...'):
                        try:
                            result, confidence = classify_image(temp_file_path, classification_model)
                            
                            # Display the classification result with confidence
                            st.markdown(f"""
                            <div style='padding: 20px; 
                                     border-radius: 10px; 
                                     background-color: #e8f4ea;
                                     border: 2px solid #28a745;'>
                                <h3 style='text-align: center; 
                                         color: #28a745;'>
                                    Classification Result
                                </h3>
                                <h2 style='text-align: center; 
                                         color: #28a745;'>
                                    {result}
                                </h2>
                                <p style='text-align: center; 
                                        color: #28a745;'>
                                    Confidence: {confidence:.2f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Classification error: {str(e)}")
                
                # Cleanup temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    else:  # Vehicle Counting
        st.header("Vehicle Counting in Video")
        
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        # Handle new video upload
        if handle_video_upload(uploaded_file):
            st.success("New video uploaded successfully!")
        
        if uploaded_file is not None:
            # Video processing controls
            col1, col2 = st.columns([1, 1])
            with col1:
                start_button = st.button("Start Processing")
            with col2:
                stop_button = st.button("Stop Processing")

            # Initialize detector
            detector = VehicleDetectionSystem(confidence_threshold=0.25)
            
            # Create placeholders
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            try:
                if start_button:
                    st.session_state.processing = True
                    # Initialize video capture
                    if st.session_state.cap is None:
                        st.session_state.cap = cv2.VideoCapture(st.session_state.temp_file_path)
                
                if stop_button:
                    st.session_state.processing = False
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                
                if st.session_state.processing and st.session_state.cap is not None:
                    if not st.session_state.cap.isOpened():
                        st.error("Error: Could not open video file")
                    else:
                        while st.session_state.cap.isOpened() and st.session_state.processing:
                            ret, frame = st.session_state.cap.read()
                            if not ret:
                                st.session_state.processing = False
                                st.session_state.cap.release()
                                st.session_state.cap = None
                                break
                            
                            # Process frame
                            processed_frame = detector.process_frame(frame)
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Update frame
                            frame_placeholder.image(processed_frame_rgb)
                            
                            # Update vehicle count
                            st.session_state.vehicle_count = len(detector.counted_vehicles)
                            
                            # Update metrics
                            metrics_placeholder.markdown(
                                f"""
                                <div style='padding: 20px; 
                                         border-radius: 10px; 
                                         background-color: #f0f2f6; 
                                         margin: 10px 0;
                                         border: 2px solid #1f77b4;'>
                                    <h2 style='text-align: center; 
                                             color: #1f77b4; 
                                             margin: 0;'>
                                        Vehicles Detected: {st.session_state.vehicle_count}
                                    </h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            time.sleep(0.1)
                
                # Display final count
                elif st.session_state.vehicle_count > 0:
                    metrics_placeholder.markdown(
                        f"""
                        <div style='padding: 20px; 
                                 border-radius: 10px; 
                                 background-color: #f0f2f6; 
                                 margin: 10px 0;
                                 border: 2px solid #1f77b4;'>
                            <h2 style='text-align: center; 
                                     color: #1f77b4; 
                                     margin: 0;'>
                                Final Vehicles Detected: {st.session_state.vehicle_count}
                            </h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.session_state.processing = False
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
        
        # Cleanup on session end
        if uploaded_file is None and st.session_state.temp_file_path is not None:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            try:
                os.unlink(st.session_state.temp_file_path)
                st.session_state.temp_file_path = None
                st.session_state.current_file_name = None
            except:
                pass

if __name__ == "__main__":
    main()