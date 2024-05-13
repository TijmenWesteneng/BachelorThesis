import os

from CNN.test_corruptions import test_corruptions

models_dir = "outputs"
corruptions_path = "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted/test"
clean_path = "../archive/HAM10000_ordered_224_0.8_0.2/test"
csv_dir = "tests"

for file_name in os.listdir(models_dir):
    if ".pt" in file_name:
        model_path = os.path.join(models_dir, file_name)
        test_corruptions(model_path, corruptions_path, clean_path, csv_dir)
