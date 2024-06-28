import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string

csv_files = [
    "../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/mean_HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    "../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/mean_HAM10000_ordered_224_0.8_0.2_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    "../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/mean_HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    "../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/mean_HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv"
]

labels = [
    "No corrupted training data",
    "No corrupted training data (CLAHE)",
    "S = 5 & cr = 0.5",
    "S = 5 & cr = 0.5 (CLAHE)"
]

fig = plt.figure(figsize=(28, 8))
width = 0.9 / len(csv_files)

for i, csv_file in enumerate(csv_files):
    # Open the dataframe and drop the 'clean error rate' column
    df = pd.read_csv(csv_file, index_col=0)
    df.drop('clean', axis='columns')

    # Calculate the mean of each column
    df_mean = df.mean().round(2)

    # Get indexes of each x tick and plot depending on difference
    x_axis = np.arange(len(df.columns))
    bar_plot = plt.bar(x_axis + (width * i - width), df_mean, width=width, label=labels[i])
    plt.bar_label(bar_plot, padding=3)

    # Remove all _ and make each first letter capital of x tick titles
    x_titles = []
    for title in df.columns.values:
        x_titles.append(string.capwords(title.replace("_", " ")))

    plt.xticks(x_axis, x_titles)

plt.xlabel("Corruptions", fontsize=18)
plt.ylabel("Error rates", fontsize=18)
plt.title("Average error rates per corruption", fontsize=25)
plt.legend()
plt.show()
