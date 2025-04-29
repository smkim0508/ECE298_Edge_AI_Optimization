import os
val_dir = 'datasets/coco8/images/val'
image_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]

for image_path in image_files:
    test_img = cv2.imread(image_path)
    if test_img is None:
        print(f"Failed to load image: {image_path}")
        continue
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    input_img = preprocess(test_img).unsqueeze(0)  # (1, 3, H, W)

    # Evaluate the model on the input image
    orig_output = forward_output(base_model, input_img)
    # Proceed with your quantization and evaluation steps