import random
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define parameters that are the same for every model
BATCH_SIZE = 64

data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]
                                     )])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Code starts
clahe_model_paths = [
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
]

nonclahe_model_paths = [
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt",
    "../remote_development/outputs/20_epochs_3_earlystopping_32_batch/HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_model.pt"
]

clahe_test_path = "../archive/HAM10000_ordered_224_0.8_0.2_clahe/test"
nonclahe_test_path = "../archive/HAM10000_ordered_224_0.8_0.2/test"

def test_model(model_path, dataset_path):
    model = torch.load(model_path)
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

    return test_data, np.array(preds)


def get_wrongly_classified(wrongly_classified_clahe_bool_list: list, wrongly_classified_nonclahe_bool_list: list):
    wrongly_classified_clahe_i_list = []
    rightly_classified_clahe_i_list = []
    for i, list in enumerate(wrongly_classified_clahe_bool_list):
        wrongly_classified_clahe_i_list.append([])
        rightly_classified_clahe_i_list.append([])
        for j, bool in enumerate(list):
            if bool:
                wrongly_classified_clahe_i_list[i].append(j)
            else:
                rightly_classified_clahe_i_list[i].append(j)

    wrongly_classified_nonclahe_i_list = []
    rightly_classified_nonclahe_i_list = []
    for i, list in enumerate(wrongly_classified_nonclahe_bool_list):
        wrongly_classified_nonclahe_i_list.append([])
        rightly_classified_nonclahe_i_list.append([])
        for j, bool in enumerate(list):
            if bool:
                wrongly_classified_nonclahe_i_list[i].append(j)
            else:
                rightly_classified_nonclahe_i_list[i].append(j)

    wrong_in_all_clahe = []
    for wrong_indice in wrongly_classified_clahe_i_list[0]:
        for indice_list in wrongly_classified_clahe_i_list:
            if wrong_indice in indice_list:
                if wrong_indice not in wrong_in_all_clahe:
                    wrong_in_all_clahe.append(wrong_indice)
            else:
                if wrong_indice in wrong_in_all_clahe:
                    wrong_in_all_clahe.remove(wrong_indice)

    wrong_in_all_nonclahe = []
    for wrong_indice in wrongly_classified_nonclahe_i_list[0]:
        for indice_list in wrongly_classified_nonclahe_i_list:
            if wrong_indice in indice_list:
                if wrong_indice not in wrong_in_all_nonclahe:
                    wrong_in_all_nonclahe.append(wrong_indice)
            else:
                if wrong_indice in wrong_in_all_nonclahe:
                    wrong_in_all_nonclahe.remove(wrong_indice)

    right_in_all_clahe = []
    for right_indice in rightly_classified_clahe_i_list[0]:
        for indice_list in rightly_classified_clahe_i_list:
            if right_indice in indice_list:
                if right_indice not in right_in_all_clahe:
                    right_in_all_clahe.append(right_indice)
            else:
                if right_indice in right_in_all_clahe:
                    right_in_all_clahe.remove(right_indice)

    right_in_all_nonclahe = []
    for right_indice in rightly_classified_nonclahe_i_list[0]:
        for indice_list in rightly_classified_nonclahe_i_list:
            if right_indice in indice_list:
                if right_indice not in right_in_all_nonclahe:
                    right_in_all_nonclahe.append(right_indice)
            else:
                if right_indice in right_in_all_nonclahe:
                    right_in_all_nonclahe.remove(right_indice)

    wrong_only_clahe = []
    for wrong_indice in wrong_in_all_clahe:
        if wrong_indice in right_in_all_nonclahe:
            wrong_only_clahe.append(wrong_indice)

    wrong_only_nonclahe = []
    for wrong_indice in wrong_in_all_nonclahe:
        if wrong_indice in right_in_all_clahe:
            wrong_only_nonclahe.append(wrong_indice)

    return wrong_only_clahe, wrong_only_nonclahe


wrongly_classified_clahe_bool_list = []
wrongly_classified_nonclahe_bool_list = []

for clahe_model_path in clahe_model_paths:
    clahe_test_data, clahe_preds = test_model(clahe_model_path, clahe_test_path)
    wrongly_classified_clahe_bool_list.append((~np.equal(np.array(clahe_test_data.targets), np.array(clahe_preds))).astype(int))

for nonclahe_model_path in nonclahe_model_paths:
    nonclahe_test_data, nonclahe_preds = test_model(nonclahe_model_path, nonclahe_test_path)
    wrongly_classified_nonclahe_bool_list.append((~np.equal(np.array(nonclahe_test_data.targets), np.array(nonclahe_preds))).astype(int))

wrong_only_clahe, wrong_only_nonclahe = get_wrongly_classified(wrongly_classified_clahe_bool_list, wrongly_classified_nonclahe_bool_list)
print(wrong_only_clahe)
print(wrong_only_nonclahe)

random_selection_clahe = random.sample(wrong_only_clahe, 10)
random_selection_nonclahe = random.sample(wrong_only_nonclahe, 10)

fig = plt.figure(figsize=(10, 8))
fig.suptitle("Images wrongly classified by CLAHE model and rightly classified by non-CLAHE model", fontsize=15)
for i, selection in enumerate(random_selection_clahe):
    """if clahe_test_data.classes[clahe_test_data.targets[selection]] == 'nv':
        i =- 1
        continue"""
    if i > 9:
        break

    img = plt.imread(clahe_test_data.imgs[selection][0])
    img2 = plt.imread(nonclahe_test_data.imgs[selection][0])

    fig.add_subplot(5, 4, i * 2 + 1)
    if i < 2:
        plt.title("Non-CLAHE")
    plt.imshow(img2)
    plt.text(25, 200, clahe_test_data.classes[clahe_test_data.targets[selection]], fontdict=dict(color="green", size="large"))
    plt.axis('off')

    fig.add_subplot(5, 4, i * 2 + 2)
    if i < 2:
        plt.title("CLAHE")
    plt.imshow(img)
    plt.text(25, 200, clahe_test_data.classes[clahe_preds[selection]], fontdict=dict(color="red", size="large"))
    plt.axis('off')

plt.show()

fig = plt.figure(figsize=(10, 8))
fig.suptitle("Images rightly classified by CLAHE model and wrongly classified by non-CLAHE model", fontsize=15)
for i, selection in enumerate(random_selection_nonclahe):
    """if nonclahe_test_data.classes[nonclahe_test_data.targets[selection]] == 'nv':
        i =- 1
        continue"""
    if i > 9:
        break

    img = plt.imread(clahe_test_data.imgs[selection][0])
    img2 = plt.imread(nonclahe_test_data.imgs[selection][0])

    fig.add_subplot(5, 4, i * 2 + 1)
    if i < 2:
        plt.title("Non-CLAHE")
    plt.imshow(img2)
    plt.text(25, 200, clahe_test_data.classes[nonclahe_preds[selection]], fontdict=dict(color="red", size="large"))
    plt.axis('off')

    fig.add_subplot(5, 4, i * 2 + 2)
    if i < 2:
        plt.title("CLAHE")
    plt.imshow(img)
    plt.text(25, 200, clahe_test_data.classes[clahe_preds[selection]], fontdict=dict(color="green", size="large"))
    plt.axis('off')

plt.show()





