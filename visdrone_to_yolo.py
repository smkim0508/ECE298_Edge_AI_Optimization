import os
import shutil

def visdrone_to_yolo(input_dir, output_dir, image_dir, classes_to_include=None):
   os.makedirs(output_dir, exist_ok=True)
   os.makedirs(image_dir, exist_ok=True)

   annotation_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
   skipped_files = 0
   total_labels = 0

   for ann_file in annotation_files:
      ann_path = os.path.join(input_dir, ann_file)
      label_lines = []
      
      with open(ann_path, 'r') as f:
         for line in f:
               parts = line.strip().split(',')
               if len(parts) < 8:
                  continue  # Skip incomplete lines
               if '' in parts:
                  continue  # Skip lines with missing fields

               try:
                  x, y, w, h, score, obj_cls, trunc, occ = map(float, parts)
               except ValueError:
                  continue  # Skip if parsing fails

               if score == 0:  # Skip if detection confidence is 0
                  continue

               obj_cls = int(obj_cls) - 1  # VisDrone is 1-indexed, YOLO expects 0-indexed

               if classes_to_include and obj_cls not in classes_to_include:
                  continue  # If filtering classes, skip others

               # Bounding box center (x,y), width, height
               x_center = (x + w / 2)
               y_center = (y + h / 2)

               # Assume standard VisDrone image size (1920x1080)
               img_width = 1920
               img_height = 1080

               # Normalize to [0, 1]
               x_center /= img_width
               y_center /= img_height
               w /= img_width
               h /= img_height

               label_lines.append(f"{obj_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

      # Write YOLO label file
      if label_lines:
         output_label_path = os.path.join(output_dir, ann_file)
         with open(output_label_path, 'w') as f_out:
               f_out.write('\n'.join(label_lines))
         total_labels += 1
      else:
         skipped_files += 1

def copy_images(src_dir, dest_dir):
   os.makedirs(dest_dir, exist_ok=True)
   image_files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
   for img_file in image_files:
      shutil.copy(os.path.join(src_dir, img_file), os.path.join(dest_dir, img_file))

if __name__ == "__main__":
   # Paths
   base_dir = 'datasets/visdrone'
   output_images_dir = os.path.join(base_dir, 'images')
   output_labels_dir = os.path.join(base_dir, 'labels')

   # Train set
   print("Processing training set...")
   visdrone_to_yolo(
      input_dir=os.path.join(base_dir, 'VisDrone2019-DET-train', 'annotations'),
      output_dir=os.path.join(output_labels_dir, 'train'),
      image_dir=os.path.join(output_images_dir, 'train')
   )
   copy_images(
      src_dir=os.path.join(base_dir, 'VisDrone2019-DET-train', 'images'),
      dest_dir=os.path.join(output_images_dir, 'train')
   )

   # Validation set
   print("Processing validation set...")
   visdrone_to_yolo(
      input_dir=os.path.join(base_dir, 'VisDrone2019-DET-val', 'annotations'),
      output_dir=os.path.join(output_labels_dir, 'val'),
      image_dir=os.path.join(output_images_dir, 'val')
   )
   copy_images(
      src_dir=os.path.join(base_dir, 'VisDrone2019-DET-val', 'images'),
      dest_dir=os.path.join(output_images_dir, 'val')
   )

   print("Conversion complete! ✅")
