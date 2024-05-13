from PIL import Image, ImageOps
import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

main_src = "../archive_nogit/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5/train+val_corrupted"
main_des = "../archive/train+val/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented/train+val"
exclude = ["nv"]


def augment_folder(src, des):
    for file in tqdm(os.listdir(src)):
        file_dir = os.path.join(src, file)
        img = Image.open(file_dir)

        img_res = img.resize((224, 224), resample=Image.BILINEAR)

        for deg in range(0, 180, 90):
            # Rotate the image and save it
            img_rot = img_res.rotate(deg)
            img_rot_file_name = os.path.join(des, Path(file).stem + "_" + str(deg))
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


def augment_folders(src, des, exclude):
    if not os.path.exists(des):
        os.makedirs(des)
    for folder in tqdm(os.listdir(src)):
        src_fol_dir = os.path.join(src, folder)
        des_fol_dir = os.path.join(des, folder)
        if not os.path.exists(des_fol_dir):
            os.mkdir(des_fol_dir)

        if folder in exclude:
            for file in tqdm(os.listdir(src_fol_dir)):
                copyfile(os.path.join(src_fol_dir, file), os.path.join(des_fol_dir, file))
        else:
            augment_folder(src_fol_dir, des_fol_dir)


if __name__ == "__main__":
    augment_folders(main_src, main_des, exclude)