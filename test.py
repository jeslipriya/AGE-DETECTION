"""
Age Detection System using Deep Learning

Description:
This system uses OpenCV with pre-trained deep learning models to:
1. Detect faces in images or real-time video
2. Predict the age group of detected faces
3. Display results with bounding boxes and age labels

Key Features:
- Supports both image files and real-time webcam input
- Uses OpenCV's DNN module for efficient inference
- Provides confidence scores for predictions
- Clean visualization of results
"""

import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime

class AgeDetectionSystem:
    """Main class for the Age Detection System"""
    
    def __init__(self):
        # Age groups defined by the model
        self.AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                         '(38-43)', '(48-53)', '(60-100)']
        
        # Model configuration
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Initialize models
        self.face_net = None
        self.age_net = None
        
        # Performance metrics
        self.frame_count = 0
        self.total_inference_time = 0
        self.start_time = None
    
    def load_models(self, face_proto, face_model, age_proto, age_model):
        """
        Load face detection and age prediction models
        
        Args:
            face_proto: Path to face detection prototxt
            face_model: Path to face detection model weights
            age_proto: Path to age prediction prototxt  
            age_model: Path to age prediction model weights
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        print("\n[SYSTEM] Loading models...")
        
        try:
            # Load face detection model (TensorFlow)
            self.face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
            
            # Load age prediction model (Caffe)
            self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
            
            # Print model information
            print(f"[SUCCESS] Face detection model loaded: {face_model}")
            print(f"[SUCCESS] Age prediction model loaded: {age_model}")
            
            # Set preferable backends (optimization for hackathon demo)
            self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return False
    
    def detect_faces(self, frame, confidence_threshold=0.7):
        """
        Detect faces in a frame using the face detection model
        
        Args:
            frame: Input image/frame
            confidence_threshold: Minimum confidence to consider a detection
            
        Returns:
            list: Bounding boxes of detected faces [x1, y1, x2, y2]
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Create blob and perform forward pass
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Process detections
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Calculate box coordinates
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, 
                                                         frame_width, frame_height])
                face_boxes.append(box.astype("int"))
                
        return face_boxes
    
    def predict_age(self, face_img):
        """
        Predict age group from a face image
        
        Args:
            face_img: Cropped face image
            
        Returns:
            tuple: (age_group, confidence)
        """
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                    self.MODEL_MEAN_VALUES, swapRB=False)
        
        # Perform inference
        start_time = time.time()
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        inference_time = time.time() - start_time
        
        # Track performance metrics
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        # Get prediction with highest confidence
        age_idx = age_preds[0].argmax()
        return self.AGE_GROUPS[age_idx], age_preds[0][age_idx]
    
    def process_frame(self, frame, display_stats=False):
        """
        Process a single frame - detect faces and predict ages
        
        Args:
            frame: Input frame
            display_stats: Whether to display performance statistics
            
        Returns:
            Processed frame with annotations
        """
        # Make copy of original frame
        output_frame = frame.copy()
        
        # Detect faces
        face_boxes = self.detect_faces(frame)
        
        # Process each detected face
        for box in face_boxes:
            x1, y1, x2, y2 = box
            
            # Extract face region
            face = frame[max(0, y1):min(y2, frame.shape[0]), 
                       max(0, x1):min(x2, frame.shape[1])]
            
            if face.size == 0:
                continue
                
            # Predict age
            age, confidence = self.predict_age(face)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"Age: {age} ({confidence*100:.1f}%)"
            
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(output_frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         (0, 255, 0), -1)
            
            # Put age text
            cv2.putText(output_frame, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display performance stats if enabled
        if display_stats and self.frame_count > 0:
            avg_inference = self.total_inference_time / self.frame_count
            fps = self.frame_count / (time.time() - self.start_time)
            
            stats = f"FPS: {fps:.1f} | Inference: {avg_inference*1000:.1f}ms"
            cv2.putText(output_frame, stats, 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output_frame
    
    def process_image(self, image_path, output_dir="output"):
        """
        Process a single image file
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save processed images
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return
            
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[SYSTEM] Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return
        
        # Process frame
        processed_frame = self.process_frame(frame)
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"age_detected_{timestamp}.jpg")
        cv2.imwrite(output_path, processed_frame)
        
        print(f"[SUCCESS] Result saved to: {output_path}")
        
        # Display result
        cv2.imshow('Age Detection Result', processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_video(self, video_source=0):
        """
        Process video stream (webcam or video file)
        
        Args:
            video_source: Path to video file or camera index
        """
        print("\n[SYSTEM] Starting video processing...")
        print("[INFO] Press 'q' to quit, 's' to save snapshot")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {video_source}")
            return
            
        # Initialize performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.total_inference_time = 0
        
        # Create output directory for snapshots
        os.makedirs("snapshots", exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame, display_stats=True)
            
            # Display result
            cv2.imshow('Real-time Age Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = f"snapshots/snapshot_{timestamp}.jpg"
                cv2.imwrite(snapshot_path, processed_frame)
                print(f"[INFO] Snapshot saved to: {snapshot_path}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        if self.frame_count > 0:
            duration = time.time() - self.start_time
            print(f"\n[PERFORMANCE] Processed {self.frame_count} frames in {duration:.1f} seconds")
            print(f"[PERFORMANCE] Average FPS: {self.frame_count/duration:.1f}")
            print(f"[PERFORMANCE] Average inference time: {self.total_inference_time/self.frame_count*1000:.1f}ms")

def print_banner():
    """Print a formatted banner for the presentation"""
    banner = """
    #######################################################
    #               AGE DETECTION SYSTEM                  #
    #                   Hackathon 2023                    #
    #                                                     #
    #  Uses OpenCV with pre-trained deep learning models  #
    #  to detect faces and predict age groups in real-time #
    #######################################################
    """
    print(banner)

def check_model_files(model_paths):
    """Check if required model files exist"""
    missing = [path for path in model_paths if not os.path.exists(path)]
    if missing:
        print("\n[ERROR] Missing required model files:")
        for path in missing:
            print(f"  - {path}")
        
        print("\nPlease download the following models:")
        print("1. Face Detection (TensorFlow):")
        print("   - opencv_face_detector.pbtxt")
        print("   - opencv_face_detector_uint8.pb")
        print("\n2. Age Prediction (Caffe):")
        print("   - age_deploy.prototxt")
        print("   - age_net.caffemodel")
        print("\nThese can be found in OpenCV's GitHub repository or model zoo.")
        return False
    return True

def main():
    """Main function to run the age detection system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Age Detection System for Hackathon Presentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', '--image', type=str, 
                       help='Process a single image file')
    parser.add_argument('-v', '--video', type=str, 
                       help='Process a video file (or leave blank for webcam)')
    parser.add_argument('--face_proto', type=str, default='models/opencv_face_detector.pbtxt',
                       help='Path to face detection prototxt')
    parser.add_argument('--face_model', type=str, default='models/opencv_face_detector_uint8.pb',
                       help='Path to face detection model weights')
    parser.add_argument('--age_proto', type=str, default='models/age_deploy.prototxt',
                       help='Path to age prediction prototxt')
    parser.add_argument('--age_model', type=str, default='models/age_net.caffemodel',
                       help='Path to age prediction model weights')
    
    args = parser.parse_args()
    
    # Print presentation banner
    print_banner()
    
    # Check for required model files
    model_paths = [args.face_proto, args.face_model, args.age_proto, args.age_model]
    if not check_model_files(model_paths):
        return
    
    # Initialize system
    age_detector = AgeDetectionSystem()
    
    # Load models
    if not age_detector.load_models(args.face_proto, args.face_model,
                                  args.age_proto, args.age_model):
        return
    
    # Process input based on arguments
    if args.image:
        age_detector.process_image(args.image)
    elif args.video is not None:
        # If video path provided, try to open it (or use webcam if empty string)
        video_source = args.video if args.video else 0
        age_detector.process_video(video_source)
    else:
        # Default to webcam if no input specified
        print("\n[SYSTEM] Starting webcam feed...")
        age_detector.process_video(0)

if __name__ == "__main__":
    main()