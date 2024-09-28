from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('runs/train3/weights/best.pt')
# model = YOLO('runs/train/weights/best.pt')

# Perform inference on an example image
results = model('https://ultralytics.com/images/photo_2024-09-28-14-13-01_jpeg.rf.6f3c681bbd1d1f79f7bdb76d222ef05f.jpg')

# Access the first result and display the image
results[0].show()