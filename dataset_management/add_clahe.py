import cv2
import os
from pathlib import Path
from tqdm import tqdm


def add_clahe_to_dataset(src_dir, des_dir):
    for folder in tqdm(os.listdir(src_dir)):
        des_fol_dir = os.path.join(des_dir, folder)
        if not os.path.exists(des_fol_dir):
            os.makedirs(des_fol_dir)

        src_fol_dir = os.path.join(src_dir, folder)
        for file in tqdm(os.listdir(src_fol_dir)):
            file_dir = os.path.join(src_fol_dir, file)

            img = cv2.imread(file_dir)
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            lab_img[...,0] = clahe.apply(lab_img[...,0])

            img_bgr = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

            cv2.imwrite(os.path.join(des_fol_dir, Path(file).stem + "_clahe.jpg"), img_bgr)


if __name__ == "__main__":
    add_clahe_to_dataset(
        "../archive/HAM10000_ordered_224_0.8_0.2/test",
        "../archive/HAM10000_ordered_224_0.8_0.2_clahe/test"
    )