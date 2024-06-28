import random

import numpy as np
import torch
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

model_path = "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
test_dir = "../archive/HAM10000_ordered_224_0.8_0.2_clahe/test"
model_path2 = "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
test_dir2 = "../archive/HAM10000_ordered_224_0.8_0.2/test"
model_path_clahe2 = "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_model.pt"

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

model2 = torch.load(model_path2)
testData2 = datasets.ImageFolder(root=test_dir2, transform=data_transform, target_transform=None)
testDataLoader2 = DataLoader(testData2, batch_size=BATCH_SIZE)

model_clahe2 = torch.load(model_path_clahe2)

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

targets_clahe = np.array(testData.targets)
preds_clahe = np.array(preds)
differing_clahe = (~np.equal(targets_clahe, preds_clahe)).astype(int)
differing_clahe_i = np.flatnonzero(differing_clahe)

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

wrong_clahe2 = (~np.equal(np.array(testData.targets), np.array(preds))).astype(int)
wrong_clahe2_i = np.flatnonzero(wrong_clahe2)
wrong_both_clahe = np.equal(differing_clahe, wrong_clahe2).astype(int)
wrong_both_clahe_i = np.flatnonzero(wrong_both_clahe)

# --------------------------- FROM HERE IS NON-CLAHE

with torch.no_grad():
    # Set the model in evaluation mode
    model2.eval()

    # Initialize a list to store the predictions
    preds = []

    # loop over the test set
    for (x, y) in tqdm(testDataLoader2):
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model2(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

    print(classification_report(np.array(testData.targets), np.array(preds), target_names=testData.classes))

targets_nonclahe = np.array(testData.targets)
preds_nonclahe = np.array(preds)
differing_nonclahe = (~np.equal(targets_nonclahe, preds_nonclahe)).astype(int)
differing_nonclahe_i = np.flatnonzero(differing_nonclahe)

just_wrong_clahe = []
for entry in wrong_both_clahe_i:
    if entry not in differing_nonclahe_i:
        just_wrong_clahe.append(entry)

just_wrong_nonclahe = []
for entry in differing_nonclahe_i:
    if entry not in differing_clahe_i and entry not in wrong_clahe2_i:
        just_wrong_nonclahe.append(entry)

just_wrong_clahe_nonv = []
for entry in just_wrong_clahe:
    if testData.classes[targets_clahe[entry]] != 'nv':
        just_wrong_clahe_nonv.append(entry)

just_wrong_nonclahe_nonv = []
for entry in just_wrong_nonclahe:
    if testData.classes[targets_nonclahe[entry]] != 'nv':
        just_wrong_nonclahe_nonv.append(entry)

random_selection = random.sample(just_wrong_clahe_nonv, 10)
random_selection2 = random.sample(just_wrong_nonclahe_nonv, 10)

fig = plt.figure(figsize=(10, 8))
fig.suptitle("Images wrongly classified by CLAHE model and rightly classified by non-CLAHE model", fontsize=15)
for i, selection in enumerate(random_selection):
    if testData.classes[targets_clahe[selection]] == 'nv':
        i =- 1
        continue
    if i > 9:
        break

    img = plt.imread(testData.imgs[selection][0])
    img2 = plt.imread(testData2.imgs[selection][0])

    fig.add_subplot(5, 4, i * 2 + 1)
    if i < 2:
        plt.title("Non-CLAHE")
    plt.imshow(img2)
    plt.text(25, 200, testData.classes[targets_clahe[selection]], fontdict=dict(color="green", size="large"))
    plt.axis('off')

    fig.add_subplot(5, 4, i * 2 + 2)
    if i < 2:
        plt.title("CLAHE")
    plt.imshow(img)
    plt.text(25, 200, testData.classes[preds_clahe[selection]], fontdict=dict(color="red", size="large"))
    plt.axis('off')

plt.show()

fig = plt.figure(figsize=(10, 8))
fig.suptitle("Images rightly classified by CLAHE model and wrongly classified by non-CLAHE model", fontsize=15)
for i, selection in enumerate(random_selection2):
    if testData.classes[targets_clahe[selection]] == 'nv':
        i =- 1
        continue
    if i > 9:
        break

    img = plt.imread(testData.imgs[selection][0])
    img2 = plt.imread(testData2.imgs[selection][0])

    fig.add_subplot(5, 4, i * 2 + 1)
    if i < 2:
        plt.title("Non-CLAHE")
    plt.imshow(img2)
    plt.text(25, 200, testData.classes[preds_nonclahe[selection]], fontdict=dict(color="red", size="large"))
    plt.axis('off')

    fig.add_subplot(5, 4, i * 2 + 2)
    if i < 2:
        plt.title("CLAHE")
    plt.imshow(img)
    plt.text(25, 200, testData.classes[preds_clahe[selection]], fontdict=dict(color="green", size="large"))
    plt.axis('off')

plt.show()
