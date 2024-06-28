import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import random
import string
from pathlib import Path
from tqdm import tqdm

right_imgs_csv_path = "right_imgs.csv"
wrong_imgs_csv_path = "wrong_imgs.csv"


def df_to_df(right_imgs_csv_path, wrong_imgs_csv_path):
    right_imgs_df = pd.read_csv(right_imgs_csv_path, index_col=0)
    wrong_imgs_df = pd.read_csv(wrong_imgs_csv_path, index_col=0)

    all_right_corrupted = []
    all_right_noncorrupted = []
    for index, row in tqdm(right_imgs_df.iterrows(), total=len(right_imgs_df)):
        for i, entry in tqdm(enumerate(row), total=len(row)):
            entry = literal_eval(entry)
            if "corrupted" in index:
                if i >= len(all_right_corrupted):
                    all_right_corrupted.append(entry)
                else:
                    for right_img in all_right_corrupted[i]:
                        if right_img not in entry:
                            all_right_corrupted[i].remove(right_img)
            else:
                if i >= len(all_right_noncorrupted):
                    all_right_noncorrupted.append(entry)
                else:
                    for right_img in all_right_noncorrupted[i]:
                        if right_img not in entry:
                            all_right_noncorrupted[i].remove(right_img)

    right_imgs_df.loc["all_right_corrupted"] = all_right_corrupted
    right_imgs_df.loc["all_right_noncorrupted"] = all_right_noncorrupted

    all_wrong_corrupted = []
    all_wrong_noncorrupted = []
    for index, row in tqdm(wrong_imgs_df.iterrows(), total=len(wrong_imgs_df)):
        for i, entry in tqdm(enumerate(row), total=len(row)):
            if "corrupted" in index:
                entry = literal_eval(entry)
                if i >= len(all_wrong_corrupted):
                    all_wrong_corrupted.append(entry)
                else:
                    for wrong_img in all_wrong_corrupted[i]:
                        if wrong_img not in entry:
                            all_wrong_corrupted[i].remove(wrong_img)
            else:
                if i >= len(all_wrong_noncorrupted):
                    all_wrong_noncorrupted.append(entry)
                else:
                    for wrong_img in all_wrong_noncorrupted[i]:
                        if wrong_img not in entry:
                            all_wrong_noncorrupted[i].remove(wrong_img)

    wrong_imgs_df.loc["all_wrong_corrupted"] = all_wrong_corrupted
    wrong_imgs_df.loc["all_wrong_noncorrupted"] = all_wrong_noncorrupted

    wrong_imgs_df.to_csv(f"{Path(wrong_imgs_csv_path).stem}_combined_all_corruptions.csv")
    right_imgs_df.to_csv(f"{Path(right_imgs_csv_path).stem}_combined_all_corruptions.csv")

    return right_imgs_df, wrong_imgs_df


def df_to_plt():
    right_imgs_df = pd.read_csv("right_imgs_combined_all_corruptions.csv", index_col=0)
    wrong_imgs_df = pd.read_csv("wrong_imgs_combined_all_corruptions.csv", index_col=0)

    right_img_per_corruption = []
    for i, corruption in enumerate(right_imgs_df.loc["all_right_corrupted"]):
        corruption = literal_eval(corruption)
        while True:
            right_img = random.choice(corruption)
            if str(right_img) in wrong_imgs_df.loc["all_wrong_noncorrupted"].iloc[i]:
                right_img_per_corruption.append((wrong_imgs_df.columns[i], right_img))
                break

    fig = plt.figure(figsize=(8, 10))
    fig.suptitle("Images correctly classified after corrupted training", fontsize="xx-large")
    for i, img_path in enumerate(right_img_per_corruption):
        img = plt.imread(img_path[1][0])
        fig.add_subplot(5, 3, i + 1)
        plt.title(string.capwords(img_path[0].replace("_", " ")))
        plt.imshow(img)
        plt.axis("off")

    plt.show()

    wrong_img_per_corruption = []
    for i, corruption in enumerate(right_imgs_df.loc["all_right_noncorrupted"]):
        while True:
            right_img = random.choice(literal_eval(corruption))
            if str(right_img) in wrong_imgs_df.loc["all_wrong_corrupted"].iloc[i]:
                wrong_img_per_corruption.append((wrong_imgs_df.columns[i], right_img))
                break

    fig = plt.figure(figsize=(8, 10))
    fig.suptitle("Images incorrectly classified after corrupted training", fontsize="xx-large")
    for i, img_path in enumerate(wrong_img_per_corruption):
        img = plt.imread(img_path[1][0])
        fig.add_subplot(5, 3, i + 1)
        plt.title(string.capwords(img_path[0].replace("_", " ")))
        plt.imshow(img)
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    #df_to_df(right_imgs_csv_path, wrong_imgs_csv_path)
    df_to_plt()


