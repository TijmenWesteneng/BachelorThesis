import os
import random
from shutil import copyfile
from tqdm import tqdm

main_dir = "../archive_nogit"
os.chdir(main_dir)
directory = "HAM10000_ordered_224"
train_ratio = 0.80
test_ratio = round(1 - train_ratio, 2)
new_directory = f"HAM10000_ordered_224_{train_ratio}_{test_ratio}"

if new_directory not in os.listdir():
    os.mkdir(new_directory)
    os.mkdir(os.path.join(new_directory, "train+val"))
    os.mkdir(os.path.join(new_directory, "test"))

for dir_name in os.listdir(directory):
    current_dir = os.path.join(directory, dir_name)
    files = os.listdir(current_dir)
    random.shuffle(files)  # Shuffle files randomly

    train_set_size = int(train_ratio * len(files))

    for i, file_name in tqdm(enumerate(files), total=len(files)):
        f = os.path.join(current_dir, file_name)

        if i < train_set_size:
            target_dir = os.path.join(new_directory, "train+val", dir_name)
        else:
            target_dir = os.path.join(new_directory, "test", dir_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        copyfile(f, os.path.join(target_dir, file_name))