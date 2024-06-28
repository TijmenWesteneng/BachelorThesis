import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd


def test_corruption(test_path, model, data_transform, batch_size, device):
    # Create the dataloader that will provide the model with data
    test_data = datasets.ImageFolder(root=test_path, transform=data_transform, target_transform=None)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()

        # Initialize a list to store the predictions
        preds = []

        # Loop over the test set
        for (x, y) in tqdm(test_dataloader):

            # Send the input to the device
            (x, y) = (x.to(device), y.to(device))

            # Make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

        return list(test_data.targets), list(preds)


def test_corruptions(model_path, corruptions_folder_path, clean_folder_path, output_csv_dir):
    if not os.path.exists(output_csv_dir):
        raise Exception(f"Output csv folder doesn't exist")

    model_name = model_path.replace("_model.pt", "")
    # Remove all the directories from the model_path so all that's left is the model name
    while model_name.find("\\") != -1 or model_name.find("/") != -1:
        if model_name.find("\\") != -1:
            model_name = model_name[model_name.find("\\") + 1:]
        if model_name.find("/") != -1:
            model_name = model_name[model_name.find("/") + 1:]
    csv_path = f"{output_csv_dir}/{model_name}_confusion.csv"
    batch_size = 64

    if os.path.exists(csv_path):
        print(f"CSV already exists: {csv_path}")
        print("Skipping testing")
        return

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]
                                         )])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)

    # Create dataframe to save all error rates in
    conf_df = pd.DataFrame()

    # Loop over all the severity folders
    severities = os.listdir(corruptions_folder_path)
    for severity in severities:
        sev_path = os.path.join(corruptions_folder_path, severity)
        # Loop over all the corruption folders and save the balanced error rates
        conf_list = []

        for corruption in os.listdir(sev_path):
            # Create corruption column in df if it doesn't exist yet
            if corruption not in conf_df.columns:
                conf_df[corruption] = []
            # Run the test on the specified corruption
            sev_cor_path = os.path.join(sev_path, corruption)
            print(f"Testing {severity}:{corruption}")
            ground_truths, predictions = test_corruption(sev_cor_path, model, data_transform, batch_size, device)
            # Calculate balanced error by subtracting balanced accuracy from 1 and rounding to 3 digits
            conf_list.append((ground_truths, predictions))

        # Add the list of balanced errors as a row to the dataframe
        conf_df.loc[len(conf_df)] = conf_list

    """
    # Calculate the clean balanced error rate and add as a column to the dataframe
    ground_truths, predictions = test_corruption(clean_folder_path, model, data_transform, batch_size, device)
    conf_df.insert(0, "clean", (ground_truths, predictions))
    """

    print(conf_df)

    # Create a dictionary with the severities as index to rename the rows of the dataframe
    index_dict = {index: value for index, value in enumerate(severities)}
    conf_df = conf_df.rename(index=index_dict)

    # Save the dataframe as a csv in the tests folder
    conf_df.to_csv(csv_path)


if __name__ == "__main__":
    test_corruptions("outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
                     "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted/test",
                     "../archive/HAM10000_ordered_224_0.8_0.2/test",
                     "tests/confusion")
