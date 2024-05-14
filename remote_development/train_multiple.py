# set the matplotlib backend so figures can be saved in the background
import os.path

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
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

train_datasets_dir = "../archive/train+val"


def train_model(dataset_name):
    # define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 20
    early_stopping_th = 3
    # define the train and val splits
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # Define the train and test directories
    train_dir = f"../archive/train+val/{dataset_name}/train+val"

    # Define saving paths
    output_path = "outputs"
    model_name = f"{dataset_name}_{EPOCHS}epochs_{early_stopping_th}early_{BATCH_SIZE}batch_{INIT_LR}lr_{TRAIN_SPLIT}train"
    model_save = f"{output_path}/{model_name}_model.pt"
    plot_save = f"{output_path}/{model_name}_plot.png"

    # Fail guard to prevent program from crashing after training
    if not os.path.exists(output_path):
        raise Exception(f"Output folder doesn't exist")

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # initialize the resnet model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_ftrs, 7))]))
    model.fc = classifier

    # Send the model to the gpu
    model.to(device)

    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)
    weights = torch.Tensor([0.382, 0.243, 0.117, 1.21, 0.125, 0.166, 1]).to(device)
    lossFn = nn.NLLLoss(weights)
    softMax = nn.LogSoftmax(dim=1)
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # Keep track of the best loss for early stopping
    lowest_val_loss = float("inf")
    epochs_without_improvement = 0
    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()

    # loop over our epochs
    for e in range(0, EPOCHS):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in tqdm(trainDataLoader):
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(softMax(pred), y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in valDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(softMax(pred), y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

        # Early stopping: Check if validation loss has improved, if so save the model, if not improved for th: break
        if avgValLoss > lowest_val_loss:
            epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= early_stopping_th:
                print(f"Early stopping at {e} epochs")
                break
        else:
            epochs_without_improvement = 0
            best_model = model

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss and accuracy
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

    # serialize the model to disk
    torch.save(best_model, model_save)


# Loop over all the dataset directories in the folder and train a model with each of them
for dir in os.listdir(train_datasets_dir):
    train_model(dir)