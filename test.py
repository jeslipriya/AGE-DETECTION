import cv2
import numpy as np
import argparse
import os

class AgeDetector:
    def __init__(self):
        # Age groups defined by the model
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                        '(38-43)', '(48-53)', '(60-100)']
        
        # Paths to model files (you'll need to download these)
        self.age_net = None
        self.face_net = None
        
        # Model configuration
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
    def load_models(self, face_proto_path, face_model_path, age_proto_path, age_model_path):
        """Load the face detection and age prediction models"""
        try:
            # Load face detection model
            self.face_net = cv2.dnn.readNetFromTensorflow(face_model_path, face_proto_path)
            
            # Load age prediction model  
            self.age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_face_box(self, net, frame, conf_threshold=0.7):
        """Detect faces in the frame"""
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123])
        
        # Set input to the model
        net.setInput(blob)
        
        # Run forward pass
        detections = net.forward()
        
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
                
        return face_boxes
    
    def predict_age(self, face_img):
        """Predict age from face image"""
        # Prepare the face image for age prediction
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        
        # Set input and run forward pass
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        # Get the age group with highest probability
        age_idx = age_preds[0].argmax()
        age = self.age_list[age_idx]
        confidence = age_preds[0][age_idx]
        
        return age, confidence
    
    def detect_age_from_image(self, image_path):
        """Detect age from a single image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
            
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not load image: {image_path}")
            return
            
        # Get face detections
        face_boxes = self.get_face_box(self.face_net, frame)
        
        if not face_boxes:
            print("No faces detected in the image")
            return
            
        # Process each detected face
        for i, face_box in enumerate(face_boxes):
            # Extract face region
            x1, y1, x2, y2 = face_box
            face = frame[max(0, y1):min(y2, frame.shape[0]), 
                        max(0, x1):min(x2, frame.shape[1])]
            
            if face.size == 0:
                continue
                
            # Predict age
            age, confidence = self.predict_age(face)
            
            # Draw bounding box and age prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add age text
            label = f"Age: {age} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Display result
        cv2.imshow('Age Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_path = f"age_detected_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, frame)
        print(f"Result saved as: {output_path}")
    
    def detect_age_from_webcam(self):
        """Real-time age detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get face detections
            face_boxes = self.get_face_box(self.face_net, frame)
            
            # Process each detected face
            for face_box in face_boxes:
                x1, y1, x2, y2 = face_box
                face = frame[max(0, y1):min(y2, frame.shape[0]), 
                           max(0, x1):min(x2, frame.shape[1])]
                
                if face.size == 0:
                    continue
                    
                # Predict age
                age, confidence = self.predict_age(face)
                
                # Draw bounding box and age prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add age text
                label = f"Age: {age}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Real-time Age Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def download_models():
    """Instructions for downloading required models"""
    print("\nTo use this age detection system, you need to download the following model files:")
    print("\n1. Face Detection Models:")
    print("   - opencv_face_detector.pbtxt")
    print("   - opencv_face_detector_uint8.pb")
    print("\n2. Age Prediction Models:")
    print("   - age_deploy.prototxt")  
    print("   - age_net.caffemodel")
    print("\nYou can download these from:")
    print("https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")
    print("https://github.com/GilLevi/AgeGenderDeepLearning")
    print("\nAlternatively, search for 'OpenCV age gender detection models' online.")

def main():
    parser = argparse.ArgumentParser(description='Age Detection using OpenCV')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--face_proto', type=str, default='opencv_face_detector.pbtxt',
                       help='Path to face detection prototxt file')
    parser.add_argument('--face_model', type=str, default='opencv_face_detector_uint8.pb',
                       help='Path to face detection model file')
    parser.add_argument('--age_proto', type=str, default='age_deploy.prototxt',
                       help='Path to age prediction prototxt file')
    parser.add_argument('--age_model', type=str, default='age_net.caffemodel',
                       help='Path to age prediction model file')
    
    args = parser.parse_args()
    
    # Check if model files exist
    model_files = [args.face_proto, args.face_model, args.age_proto, args.age_model]
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing model files: {missing_files}")
        download_models()
        return
    
    # Initialize age detector
    detector = AgeDetector()
    
    # Load models
    if not detector.load_models(args.face_proto, args.face_model, 
                               args.age_proto, args.age_model):
        print("Failed to load models!")
        return
    
    # Run detection based on arguments
    if args.webcam:
        print("Starting webcam age detection. Press 'q' to quit.")
        detector.detect_age_from_webcam()
    elif args.image:
        detector.detect_age_from_image(args.image)
    else:
        print("Please specify either --image or --webcam")
        print("Example usage:")
        print("  python age_detection.py --image photo.jpg")
        print("  python age_detection.py --webcam")

if __name__ == "__main__":
    main()