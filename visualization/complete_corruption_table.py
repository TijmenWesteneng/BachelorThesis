import pandas as pd
import os

df_paths = {
    '0': "pct_mean_HAM10000_ordered_224_0.8_0.2_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '0C': "pct_mean_HAM10000_ordered_224_0.8_0.2_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '1': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '1C': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s1_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '2': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '2C': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s2_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '3': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '3C': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s3_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '4': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '4C': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s4_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '5': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_20epochs_3early_32batch_0.001lr_0.8train_test.csv",
    '5C': "pct_mean_HAM10000_ordered_224_0.8_0.2_corrupted_s5_cr0.5_augmented_clahe_20epochs_3early_32batch_0.001lr_0.8train_test.csv"
}

os.chdir("../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/percentages")

dfs = {}

for key in df_paths:
    dfs[key] = pd.read_csv(df_paths[key], index_col=0)

df_table = pd.DataFrame(columns=dfs[key].columns)

for sev in range(1, 6):
    for key in dfs:
        df_table.loc[f"{key}-{sev}"] = dfs[key].loc[sev]

df_table.to_csv("pct_corruption_table_all.csv")