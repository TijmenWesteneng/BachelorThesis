import os


def check_dataset(dataset_folder_path):
    corruption_types = ['none', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                        'motion_blur', 'zoom_blur', 'brightness_down', 'brightness', 'contrast',
                        'black_corner', 'characters', 'elastic_transform', 'pixelate', 'jpeg_compression']

    corruption_dict = dict.fromkeys(corruption_types, 0)

    dataset_folder_path = os.path.join(dataset_folder_path, "train+val")

    for class_folder in os.listdir(dataset_folder_path):
        dataset_folder_class_path = os.path.join(dataset_folder_path, class_folder)
        for file_name in os.listdir(dataset_folder_class_path):
            corrupt = False
            for corruption_type in corruption_types:
                if corruption_type in file_name:
                    corruption_dict[corruption_type] += 1
                    corrupt = True
                    break
            if not corrupt:
                corruption_dict['none'] += 1

    return corruption_dict


def check_multiple(folder):
    for dataset in os.listdir(folder):
        dataset_folder = os.path.join(folder, dataset)
        print(f"Checking for: {dataset}")
        corruption_dict = check_dataset(dataset_folder)
        total_corrupt = 0
        for key in corruption_dict.keys():
            if key != "none":
                total_corrupt += corruption_dict[key]
        print(f"Non-corrupt: {corruption_dict['none']}; Corrupt: {total_corrupt}; "
              f"Corruption-ratio: {total_corrupt / (corruption_dict['none'] + total_corrupt)}")
        print(corruption_dict)


if __name__ == "__main__":
    check_multiple("../archive/train+val")