import os

from CNN.test_corruptions import test_corruptions

models_dir = "outputs"
corruptions_path = "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted/test"
clean_path = "../archive/HAM10000_ordered_224_0.8_0.2/test"
csv_dir = "tests"

for file_name in os.listdir(models_dir):
    model_path = os.path.join(models_dir, file_name)
    # Check if file is not a directory and if it is a model (.pt extension)
    if os.path.isfile(model_path) and ".pt" in file_name:
        print(f"Testing corruptions for: {model_path}")
        test_corruptions(model_path, corruptions_path, clean_path, csv_dir)
