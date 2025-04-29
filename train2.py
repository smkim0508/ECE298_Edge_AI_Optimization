from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on COCO8
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

model.save("yolo11n_model.pt")
print("Training complete. Model saved as 'yolo11n.pt'.")