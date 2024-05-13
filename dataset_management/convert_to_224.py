import os
from PIL import Image
from pathlib import Path

src_dir = "../archive_nogit/HAM10000_ordered/"
des_dir = "../archive_nogit/HAM10000_ordered_224"

for folder in os.listdir(src_dir):
    des_fol_dir = os.path.join(des_dir, folder)
    if not os.path.exists(des_fol_dir):
        os.mkdir(des_fol_dir)

    src_fol_dir = os.path.join(src_dir, folder)
    for file in os.listdir(src_fol_dir):
        file_dir = os.path.join(src_fol_dir, file)
        img = Image.open(file_dir)

        img_res = img.resize((224, 224), resample=Image.BILINEAR)

        img_res.save(os.path.join(des_fol_dir, Path(file).stem + "_224.jpg"))