# Set the matplotlib backend so figures can be saved in the background
import os.path

import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import torch
import time
import math
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import argparse

train_datasets_dir = "../archive/train+val"
dataset_to_train = ""
output_path = "outputs"

parser = argparse.ArgumentParser(
    description="Train multiple models (RN18 or RN50) on multiple datasets from a folder"
)

parser.add_argument("--RN50",
                    help="Train an RN50 model instead of an RN18", action="store_true")

parser.add_argument("--datasets_folder", default=train_datasets_dir, type=str,
                    help="Folder where the train datasets are")

parser.add_argument("--dataset", default=dataset_to_train, type=str,
                    help="Specific dataset to train on (leave empty if training on all datasets from folder)")

parser.add_argument("--output_path", default=output_path, type=str,
                    help="Path where to save the trained models as .pt files")

args = parser.parse_args()

train_datasets_dir = args.datasets_folder
dataset_to_train = args.dataset
output_path = args.output_path


def train_model(dataset_name):
    # Define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 20
    early_stopping_th = 3
    # Define the train and val splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # Define the train and test directories
    train_dir = os.path.join(train_datasets_dir, dataset_name, "train+val")

    # Fail guard to prevent program from crashing after training
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set the device that will be used to train the model (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard data transform that was also used to train on ImageNet for pre-trained weights
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]
                                         )
                                         ])

    # Load the skin dataset
    print("[INFO] Loading the skin dataset...")
    print(f"Train_dataset_dir: {train_dir}")
    trainData = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)

    # Calculate the train/validation split
    print("[INFO] Generating the train/validation split...")
    numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
    numValSamples = int(math.ceil(len(trainData) * VAL_SPLIT))
    (trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples],
                                        generator=torch.Generator().manual_seed(42))

    # Initialize the train & validation dataloaders
    trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)

    # Calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # Depending on args given, either init ResNet-18 or ResNet-50 model (and change model name & saving accordingly)
    if args.RN50:
        # Initialize the resnet-50 model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model_name = f"RN50_{dataset_name}_{EPOCHS}epochs_{early_stopping_th}early_{BATCH_SIZE}batch_{INIT_LR}lr_{TRAIN_SPLIT}train"
        model_save = f"{output_path}/{model_name}_model.pt"
        plot_save = f"{output_path}/{model_name}_plot.png"
    else:
        # Initialize the resnet-18 model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model_name = f"{dataset_name}_{EPOCHS}epochs_{early_stopping_th}early_{BATCH_SIZE}batch_{INIT_LR}lr_{TRAIN_SPLIT}train"
        model_save = f"{output_path}/{model_name}_model.pt"
        plot_save = f"{output_path}/{model_name}_plot.png"

    # Freeze all model parameters and add new fully connected last layer (transfer learning)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, 7))]))
    model.fc = classifier

    # Send the model to the GPU (or other device)
    model.to(device)

    # Initialize the optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)
    # Class weights to compensate for the unbalanced dataset
    weights = torch.Tensor([0.382, 0.243, 0.117, 1.21, 0.125, 0.166, 1]).to(device)
    lossFn = nn.NLLLoss(weights)
    softMax = nn.LogSoftmax(dim=1)
    # Initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # Keep track of the best loss for early stopping
    highest_balanced_accuracy = 0
    epochs_without_improvement = 0
    # Measure how long training is going to take
    print("[INFO] Training the network...")
    startTime = time.time()

    # Loop over all epochs
    for e in range(0, EPOCHS):
        # Set the model in training mode
        model.train()
        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # Initialize the number of correct predictions in the training and validation step
        trainCorrect = 0
        valCorrect = 0
        # Loop over the training set
        for (x, y) in tqdm(trainDataLoader):
            # Send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # Perform a forward pass and calculate the training loss (using softmax as input)
            pred = model(x)
            loss = lossFn(softMax(pred), y)
            # Xero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Add the loss to the total training loss so far and
            # Calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Save all predictions to compare with ground truth and calculate balanced accuracy
        valPreds = []
        valTruths = []

        # Switch off autograd for evaluation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()
            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # Make the predictions and calculate the validation loss (using softmax)
                pred = model(x)
                totalValLoss += lossFn(softMax(pred), y)
                # Calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

                # Add the predictions to the list of validation predictions and the actual labels to valTruths
                valPreds.extend(pred.argmax(axis=1).cpu().numpy())
                valTruths.extend(y.cpu().numpy())

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # Calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        # Update the training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # Print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

        bal_acc = balanced_accuracy_score(np.array(valTruths), np.array(valPreds))

        # Early stopping: Check if balanced accuracy has improved, if so save the model, if not improved for th: break
        if bal_acc < highest_balanced_accuracy:
            epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= early_stopping_th:
                print(f"Early stopping at {e + 1} epochs")
                break
        else:
            highest_balanced_accuracy = bal_acc
            epochs_without_improvement = 0
            best_model = model
            print(f"New highest validation balanced accuracy: {highest_balanced_accuracy}")

    # Finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_save)

    # Serialize the model to disk
    torch.save(best_model, model_save)


# Loop over all the dataset directories in the folder and train a model with each of them
for dir in os.listdir(train_datasets_dir):
    # Check if the directory is actually a dataset directory (useful to exclude datasets by putting them in a folder)
    if dir.find("HAM10000") != -1:
        # If dataset_to_train is empty: train on all datasets in the folder
        if len(dataset_to_train) == 0:
            train_model(dir)
        else:
            # Else: check if dataset is in dataset_to_train, otherwise don't train on it
            if dir == dataset_to_train:
                train_model(dir)