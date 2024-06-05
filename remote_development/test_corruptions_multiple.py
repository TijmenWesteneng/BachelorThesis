import os
import argparse

from CNN.test_corruptions import test_corruptions

models_dir = "outputs"
corruptions_path = "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted/test"
clean_path = "../archive/HAM10000_ordered_224_0.8_0.2/test"
csv_dir = "tests"

parser = argparse.ArgumentParser(
    description="Test corruptions of multiple trained models"
)

parser.add_argument("--corruptions_path", default=corruptions_path, type=str,
                    help="Folder where the corrupted test set is located")

parser.add_argument("--clean_path", default=clean_path, type=str,
                    help="Folder where the clean test set is located")

parser.add_argument("--models_dir", default=models_dir, type=str,
                    help="Folder where the (pretty) models are located")

parser.add_argument("--csv_dir", default=csv_dir, type=str,
                    help="Folder where the csv files will be saved")

args = parser.parse_args()

corruptions_path = args.corruptions_path
clean_path = args.clean_path
models_dir = args.models_dir
csv_dir = args.csv_dir

for file_name in os.listdir(models_dir):
    model_path = os.path.join(models_dir, file_name)
    # Check if file is not a directory and if it is a model (.pt extension)
    if os.path.isfile(model_path) and ".pt" in file_name:
        print(f"Testing corruptions for: {model_path}")
        test_corruptions(model_path, corruptions_path, clean_path, csv_dir)
