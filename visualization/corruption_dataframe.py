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

cr_df = pd.DataFrame()

for i, csv_file in enumerate(csv_files):
    # Open the dataframe and drop the 'clean error rate' column
    df = pd.read_csv(csv_file, index_col=0)
    df.drop('clean', axis='columns')

    for column in df.columns:
        if column not in cr_df.columns:
            cr_df[column] = []

    # Calculate the mean of each column
    df_mean = df.mean().round(2)
    cr_df.loc[len(cr_df)] = df_mean

index_dict = {index: value for index, value in enumerate(labels)}
cr_df = cr_df.rename(index=index_dict)
sorted_cr_df = cr_df.sort_values("No corrupted training data", axis=1, ascending=False)

first_row = cr_df.iloc[0]
pct_diff = (cr_df - first_row) / first_row * 100
sorted_pct_diff = pct_diff.sort_values("S = 5 & cr = 0.5", axis=1, ascending=True).round(2)

print(sorted_cr_df.to_string())
print(sorted_pct_diff.to_csv("pct_change.csv"))

