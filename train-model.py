from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model using your YAML file and specified parameters
model.train(
    data='data/waste/waste.yaml',
    epochs=100,
    imgsz=640,
    project='runs',
    lr0=0.01,
    lrf=0.1,
    save_period=5,
    augment=True,
)