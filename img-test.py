from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Perform inference on an example image
results = model('https://ultralytics.com/images/bus.jpg')

# Access the first result and display the image
results[0].show()