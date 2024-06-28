import numpy as np
import torch
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

model_path = "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
title = "Clean test set confusion matrix of a model \n trained on a corrupted dataset (s = 5, cr = 0.5, CLAHE)"
test_dir = "../archive/HAM10000_ordered_224_0.8_0.2_clahe/test"
BATCH_SIZE = 64

data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]
                                     )])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)
testData = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

with torch.no_grad():
    # Set the model in evaluation mode
    model.eval()

    # Initialize a list to store the predictions
    preds = []

    # loop over the test set
    for (x, y) in tqdm(testDataLoader):
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

    print(classification_report(np.array(testData.targets), np.array(preds), target_names=testData.classes))
    disp = ConfusionMatrixDisplay.from_predictions(
        np.array(testData.targets), np.array(preds), display_labels=testData.classes, normalize='true', values_format='.2f'
    )
    disp.ax_.set_title(title)
    plt.show()
