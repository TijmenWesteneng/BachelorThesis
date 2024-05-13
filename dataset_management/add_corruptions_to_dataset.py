import os
import random
from shutil import copyfile
from tqdm import tqdm

# Initialize parameters
corruption_ratio = 0.5
severity = 1
src_dat_path = "../archive/HAM10000_ordered_224_0.8_0.2/train+val"
src_cor_path = "../archive/HAM10000_ordered_224_0.8_0.2_corrupted/train+val"
des_dat_path = f"../archive/HAM10000_ordered_224_0.8_0.2_corrupted_s{severity}_cr{corruption_ratio}/train+val_corrupted"


def add_corruptions_to_dataset(src_dat: str, src_cor_dat: str, des_dat: str, cor_ratio: float, sev: int):
    """
    Takes normal and corrupted dataset and merges them into a new combined dataset that has a specified percentage
    of corrupted data in it.

    :param src_dat: string path where the source train dataset is located
    :param src_cor: string path where the corrupted train dataset is located
    :param des_dat: string path where to save the combined new dataset
    :param cor_ratio: percentage (0 - 1) of corruptions in combined new dataset
    :param sev: severity of corruptions to include
    :return: None
    """

    # Check if destination directories exist, otherwise create them
    if not os.path.exists(des_dat):
        os.makedirs(des_dat)

    # First copy the whole non-corrupted dataset into the new folder
    for folder in os.listdir(src_dat):
        src_dat_fol = os.path.join(src_dat, folder)
        des_dat_fol = os.path.join(des_dat, folder)
        if not os.path.exists(des_dat_fol):
            os.mkdir(des_dat_fol)
        for file in tqdm(os.listdir(src_dat_fol)):
            copyfile(os.path.join(src_dat_fol, file), os.path.join(des_dat_fol, file))

    src_cor_dat_sev = os.path.join(src_cor_dat, str(sev))
    corruptions = os.listdir(src_cor_dat_sev)
    for corruption in tqdm(corruptions):
        src_cor_dat_sev_cor = os.path.join(src_cor_dat_sev, corruption)
        for class_folder in os.listdir(src_cor_dat_sev_cor):
            src_cor_dat_sev_cor_class = os.path.join(src_cor_dat_sev_cor, class_folder)

            files = os.listdir(src_cor_dat_sev_cor_class)
            random.shuffle(files)

            corruption_amount_per_folder = int(round((1 / len(corruptions)) *
                                               (corruption_ratio / (1 - corruption_ratio)) * len(files)))

            for i, file_name in enumerate(files):
                if i < corruption_amount_per_folder:
                    src_file = os.path.join(src_cor_dat_sev_cor_class, file_name)
                    des_file = os.path.join(des_dat, class_folder, file_name)
                    copyfile(src_file, des_file)
                else:
                    continue


add_corruptions_to_dataset(src_dat_path, src_cor_path, des_dat_path, corruption_ratio, severity)