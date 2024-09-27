import cv2
import numpy as np
from ultralytics import YOLO
import threading

# model = YOLO('yolov8n.pt')
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

annotated_frame = None

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

        if aspect_ratio > 1:  # Wider than tall
            new_w, new_h = target_size[0], int(target_size[0] / aspect_ratio)
        else:  # Taller than wide
            new_h, new_w = target_size[1], int(target_size[1] * aspect_ratio)

        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad the resized frame to the target size
        top = (target_size[1] - new_h) // 2
        bottom = target_size[1] - new_h - top
        left = (target_size[0] - new_w) // 2
        right = target_size[0] - new_w - left
        frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT)

        # Perform inference
        annotated_frame = model(frame_padded, conf=0.4)[0].plot()

# Start a thread for processing frames
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

while True:
    if annotated_frame is not None:
        cv2.imshow('Real-time Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()