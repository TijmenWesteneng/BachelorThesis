from PIL import Image, ImageOps
import os
from pathlib import Path

directory = "../HAM10000_ordered/nv"
new_directory = "../HAM10000_augmented_2/nv/"

for file in os.listdir(directory):
    file_dir = os.path.join(directory, file)
    img = Image.open(file_dir)

    img_res = img.resize((224, 224), resample=Image.BILINEAR)

    for deg in range(0, 90, 90):
        # Rotate the image and save it
        img_rot = img_res.rotate(deg)
        img_rot_file_name = new_directory + Path(file).stem + "_" + str(deg)
        img_rot.save(img_rot_file_name + ".jpg")

        # Flip the image and save it
        img_save = ImageOps.flip(img_rot)
        img_save.save(img_rot_file_name + "_flip" + ".jpg")

        # Also mirror the image and save it
        img_save = ImageOps.mirror(img_save)
        img_save.save(img_rot_file_name + "_flip_mirror" + ".jpg")

        # Only mirror the image and save it
        img_save = ImageOps.mirror(img_rot)
        img_save.save(img_rot_file_name + "_mirror" + ".jpg")


