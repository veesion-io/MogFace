import cv2
import os
import glob
import glob
import random
input_folder = "WIDER_val/images/"
output_folder = "WIDER_val/resized_images/"
image_size = (720, 1280)  # The dimensions your model expects
num_images = 400

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

all_files = []
for subdir, _, _ in os.walk(input_folder):
    all_files.extend(glob.glob(f"{subdir}/*.jpg"))

selected_files = random.sample(all_files, num_images)

for file in selected_files:
    img = cv2.imread(file)
    resized_img = cv2.resize(img, image_size)
    output_path = file.replace(input_folder, output_folder)
    output_subdir = os.path.dirname(output_path)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    _ = cv2.imwrite(output_path, resized_img)