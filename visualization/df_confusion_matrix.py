import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import ast

df_path = "../remote_development/tests/confusion/HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_confusion.csv"
df = pd.read_csv(df_path, index_col=0)
df = df.map(ast.literal_eval)

truths = []
preds = []

for i, j in df.iterrows():
    for entry in j.values:
        for truth in entry[0]:
            truths.append(truth)
        for pred in entry[1]:
            preds.append(pred)

labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
ConfusionMatrixDisplay.from_predictions(truths, preds, normalize='true', display_labels=labels, values_format='.2f')
plt.title("Confusion matrix of all corruptions of \n a model trained on a non-corrupted dataset")
plt.show()