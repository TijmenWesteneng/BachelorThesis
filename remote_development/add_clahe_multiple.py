import os

from dataset_management.add_clahe import add_clahe_to_dataset


src_folder = "../archive/train+val"
des_folder = "../archive/train+val/clahe"

for dataset_name in os.listdir(src_folder):
    src = os.path.join(src_folder, dataset_name, "train+val")
    des = os.path.join(des_folder, dataset_name + "_clahe", "train+val")
    print(f"Adding CLAHE from: {src} to {des}")
    add_clahe_to_dataset(src, des)
