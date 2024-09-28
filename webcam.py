import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Dynamically load the model from user input
model_path = input("Enter the path to your YOLO model (e.g., 'yolov8n.pt' or 'runs/train/weights/best.pt'): ") or 'runs/train/weights/best.pt'
model = YOLO(model_path)

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Global variable to store the annotated frame for real-time display
annotated_frame = None

# Function to process each frame, apply resizing, padding, and model inference
def process_frame():
    global annotated_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame while maintaining aspect ratio
        target_size = (640, 480)
        h, w = frame.shape[:2]
        aspect_ratio = w / h

        # Calculate new width and height to fit into target_size
        if aspect_ratio > 1:  # Wider than tall
            new_w, new_h = target_size[0], int(target_size[0] / aspect_ratio)
        else:  # Taller than wide
            new_h, new_w = target_size[1], int(target_size[1] * aspect_ratio)

        # Resize the frame to fit into the target size
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # Pad the resized frame to the target size for uniform input
        top = (target_size[1] - new_h) // 2
        bottom = target_size[1] - new_h - top
        left = (target_size[0] - new_w) // 2
        right = target_size[0] - new_w - left
        frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT)

        # Perform inference using the YOLO model
        results = model(frame_padded, conf=0.4)[0]  # Perform object detection with a confidence threshold of 0.4
        annotated_frame = results.plot()  # Plot the detection results on the frame

# Start a separate thread for processing frames (to avoid blocking the main thread)
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

# Main loop for displaying the video frames with annotations
while True:
    if annotated_frame is not None:
        cv2.imshow('Real-time Object Detection', annotated_frame)  # Show the annotated frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when done
cap.release()
cv2.destroyAllWindows()