import pandas as pd
from statistics import mean

test_csv_path = "../CNN/tests/HAM10000_ordered_224_0.8_0.2_augmented_10epochs_64batch_0.001lr_0.9train_Test.csv"
base_csv_path = "../CNN/tests/RN50_HAM10000_ordered_224_0.8_0.2_augmented_10epochs_64batch_0.001lr_0.9train_Test.csv"
des_csv_path = "../CNN/tests/BCE/BCE_HAM10000_ordered_224_0.8_0.2_augmented_10epochs_64batch_0.001lr_0.9train_test.csv"

def calc_cor_err(csv_test_path, csv_base_path):
    # Initialize the dataframes needed to do the calculations
    test_df = pd.read_csv(csv_test_path)
    base_df = pd.read_csv(csv_base_path)
    err_df = pd.DataFrame()

    BCE_list = []
    rBCE_list = []
    for column in test_df.columns:
        if column == "Unnamed: 0" or column == "clean":
            continue
        BCE = sum(test_df[column]) / sum(base_df[column])
        rBCE = sum(test_df[column] - test_df["clean"]) / sum(base_df[column] - base_df["clean"])
        BCE_list.append(round(BCE, 3))
        rBCE_list.append(round(rBCE, 3))
        err_df[column] = []

    err_df.loc[len(err_df)] = BCE_list
    err_df.loc[len(err_df)] = rBCE_list

    mean_BCE = round(mean(err_df.loc[0]), 3)
    mean_rBCE = round(mean(err_df.loc[1]), 3)

    err_df.insert(0, "mean", [mean_BCE, mean_rBCE])
    err_df.insert(0, "clean error rate", test_df["clean"])

    err_df = err_df.rename({0: "BCE", 1: "rBCE"})

    print(err_df)
    return err_df


if __name__ == "__main__":
    err_df = calc_cor_err(test_csv_path, base_csv_path)
    err_df.to_csv(des_csv_path)
