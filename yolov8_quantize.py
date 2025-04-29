import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms
from glob import glob

# --- Configuration ---
input_size = (640, 640)
model_path = 'yolov8_model.pt'
coco8_val_dir = 'datasets/coco8/images/val'  # Update this path as needed

# --- Preprocessing ---
preprocess = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize(input_size),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def forward_output(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output

def compute_diff_norm(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2).item()

# --- MAIN SCRIPT EXECUTION ---
# Load the YOLOv8 model using Ultralytics.
full_model = YOLO(model_path)
base_model = full_model.model
base_model.to('cpu')
base_model.eval()

# Load all validation images from COCO8
image_paths = glob(os.path.join(coco8_val_dir, '*.jpg'))
if not image_paths:
    raise FileNotFoundError(f"No images found in '{coco8_val_dir}'.")

# Process a subset of images for calibration and evaluation
calib_image_paths = image_paths[:10]  # Use 10 images for calibration

def get_input(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = preprocess(img).unsqueeze(0)
    return input_img

# Collect original outputs
orig_outputs = []
for img_path in calib_image_paths:
    input_img = get_input(img_path)
    if input_img is None:
        continue
    output = forward_output(base_model, input_img)
    orig_outputs.append(output)

# Compute baseline (32-bit) model size.
baseline_size = sum(p.numel() * 32 for p in base_model.parameters())
print("Baseline (32-bit) results:")
print(f"  Model size: {baseline_size / 1e6:.2f} Mb (in bits)")

# --- PyTorch Static Quantization ---
import torch.quantization as tq

# 1. Fuse modules (if possible)
# YOLOv8 may not have standard blocks, so this may be a no-op
try:
    fused_model = tq.fuse_modules(base_model, [])  # No-op for custom models
except Exception:
    fused_model = base_model

# 2. Specify quantization config
fused_model.qconfig = tq.get_default_qconfig('qnnpack')

# 3. Prepare for static quantization
quant_ready_model = tq.prepare(fused_model, inplace=False)

# 4. Calibrate with a few batches
for img_path in calib_image_paths:
    input_img = get_input(img_path)
    if input_img is None:
        continue
    quant_ready_model(input_img)

# 5. Convert to quantized model
quantized_model = tq.convert(quant_ready_model, inplace=False)

# 6. Evaluate quantized model
quant_outputs = []
for img_path in calib_image_paths:
    input_img = get_input(img_path)
    if input_img is None:
        continue
    output = forward_output(quantized_model, input_img)
    quant_outputs.append(output)

# 7. Compute average difference norm
if len(orig_outputs) == len(quant_outputs) and len(orig_outputs) > 0:
    diff_norms = [compute_diff_norm(o, q) for o, q in zip(orig_outputs, quant_outputs)]
    avg_diff_norm = sum(diff_norms) / len(diff_norms)
else:
    avg_diff_norm = float('nan')

# 8. Model size (approximate, since quantized weights are int8)
quant_size = sum(p.numel() * 8 for p in quantized_model.parameters())
print("\nQuantized (8-bit) results:")
print(f"  Model size: {quant_size / 1e6:.2f} Mb (in bits)")
print(f"  Average difference vs full precision: {avg_diff_norm:.4f}")

print(torch.__version__)
