import pandas as pd
import os

from data_processing.csv_to_meancorruption import calc_cor_err

test_csv_dir = "../CNN/tests"
base_csv_path = "../CNN/tests/RN50_HAM10000_ordered_224_0.8_0.2_augmented_10epochs_64batch_0.001lr_0.9train_test.csv"
des_dir = "../CNN/tests/BCE"

for csv_file in os.listdir(test_csv_dir):
    test_csv_path = os.path.join(test_csv_dir, csv_file)
    des_csv_path = os.path.join(des_dir, f"BCE_{csv_file}")

    # Check if the current entry is a file and not a directory
    if not os.path.isfile(test_csv_path):
        continue

    # Check if current csv_file is base file and ignore if so
    if os.path.samefile(test_csv_path, base_csv_path):
        continue

    print(f"test_csv: {test_csv_path}; base_csv: {base_csv_path}")

    err_df = calc_cor_err(test_csv_path, base_csv_path)
    err_df.to_csv(des_csv_path)