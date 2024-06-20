import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Define parameters that are the same for every model
BATCH_SIZE = 64

data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]
                                     )])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Code starts
corrupted_model_paths = [
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_3/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_4/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_5/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_6/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_3/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_4/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_5/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_6/HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_3/HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_4/HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_5/HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_6/HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_3/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_4/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_5/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_6/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_3/HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_4/HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_5/HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_6/HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
]

noncorrupted_model_paths = [
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_3/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_4/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_5/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch_6/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
]

corruptions_test_path = "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted/test"

csv_save_dir = ""


def test_corruption(model, dataset_path):
    test_data = datasets.ImageFolder(root=dataset_path, transform=data_transform, target_transform=None)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()

        # Initialize a list to store the predictions
        preds = []

        # loop over the test set
        for (x, y) in tqdm(test_data_loader):
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

        print(classification_report(np.array(test_data.targets), np.array(preds), target_names=test_data.classes))

    right_prediction_array = np.equal(np.array(test_data.targets), np.array(preds))

    right_imgs = []
    wrong_imgs = []

    for i, right_prediction_array in enumerate(right_prediction_array):
        if right_prediction_array:
            right_imgs.append(test_data.imgs[i])
        else:
            wrong_imgs.append(test_data.imgs[i])

    return right_imgs, wrong_imgs


def test_corruptions(model_path, corruptions_folder_path):
    print(f"Testing for model path: {model_path}")
    model = torch.load(model_path)

    # Loop over all the severity folders
    wrongly_classified_list = []
    rightly_classified_list = []
    severities = os.listdir(corruptions_folder_path)
    for severity in severities:
        sev_path = os.path.join(corruptions_folder_path, severity)
        # Loop over all the corruption folders and save the rightly and wrongly classified imgs
        for i, corruption in enumerate(os.listdir(sev_path)):
            # Run the test on the specified corruption
            sev_cor_path = os.path.join(sev_path, corruption)
            print(f"Testing {severity}:{corruption}")
            right_imgs, wrong_imgs = test_corruption(model, sev_cor_path)
            if i > (len(wrongly_classified_list) - 1):
                wrongly_classified_list.append(wrong_imgs)
                rightly_classified_list.append(right_imgs)
            else:
                wrongly_classified_list[i].extend(wrong_imgs)
                rightly_classified_list[i].extend(right_imgs)

    return rightly_classified_list, wrongly_classified_list


def generate_dataframes(corrupted_model_paths, noncorrupted_model_paths, corruption_test_path, csv_save_dir):
    right_imgs = pd.DataFrame()
    wrong_imgs = pd.DataFrame()

    for corruption in os.listdir(os.path.join(corruption_test_path, "1")):
        right_imgs[corruption] = []
        wrong_imgs[corruption] = []

    for corrupted_model_path in corrupted_model_paths:
        rightly_classified_list, wrongly_classified_list = test_corruptions(corrupted_model_path, corruption_test_path)
        right_imgs.loc[corrupted_model_path] = rightly_classified_list
        wrong_imgs.loc[corrupted_model_path] = wrongly_classified_list

    for noncorrupted_model_path in noncorrupted_model_paths:
        rightly_classified_list, wrongly_classified_list = test_corruptions(noncorrupted_model_path, corruption_test_path)
        right_imgs.loc[noncorrupted_model_path] = rightly_classified_list
        wrong_imgs.loc[noncorrupted_model_path] = wrongly_classified_list

    right_imgs.to_csv(os.path.join(csv_save_dir, "right_imgs"))
    wrong_imgs.to_csv(os.path.join(csv_save_dir, "wrong_imgs"))


if __name__ == "__main__":
    generate_dataframes(corrupted_model_paths, noncorrupted_model_paths, corruptions_test_path, csv_save_dir)