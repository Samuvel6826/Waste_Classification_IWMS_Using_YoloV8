from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model using your custom dataset
model.train(data='data/waste/waste.yaml', epochs=5, imgsz=640, project='runs')