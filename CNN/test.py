import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

model_path = "models/firstModel.pt"
test_dir = "../archive/HAM10000_augmented_all224_0.9_0.1/test"
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
    for (x, y) in testDataLoader:
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

    print(classification_report(np.array(testData.targets), np.array(preds), target_names=testData.classes))
