# Loading Pre-trained YOLO Model
from ultralytics import YOLO

# Choose a model configuration; here we use the YOLOv8 nano version (yolov8n.yaml)
model = YOLO('yolov8n.yaml')

# Train the model.
# For demonstration, we use the built-in COCO128 dataset (which is provided by Ultralytics)
# Alternatively, change the 'data' parameter to your custom YAML file (e.g. data='cifar10_detection.yaml')
# model.train(data='coco128.yaml', epochs=20, imgsz=640, batch=16) # already pre-trained on coco, so this isn't the most helpful now

# model.train(
#     data='visdrone.yaml', 
#     epochs=5,          # fewer epochs
#     imgsz=512,         # smaller images
#     batch=8,           # smaller batch
#     cache=True         # load images into memory for faster access
# )

# Save the trained model weights.
# model.save("yolov8_drone_model.pt")
# print("Training complete. Model saved as 'yolov8_drone_model.pt'.")

model.save("yolov8_model2.pt")
print("Training complete. Model saved as 'yolov8_model2.pt'.")