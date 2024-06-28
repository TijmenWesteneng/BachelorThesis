import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

non_clahe_dir = "../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/BCE_same_baseline"
clahe_dir = "../remote_development/tests/mean_20_epochs_3_earlystopping_32_batch/BCE_same_baseline"


def plot_BCE(dir_path, clahe_dir_path = None):
    base_dir = os.getcwd()

    csv_dict = dict()
    os.chdir(dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" not in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    csv_dict_clahe = dict()
    os.chdir(base_dir)
    os.chdir(clahe_dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict_clahe["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict_clahe[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    index = list(range(0, 5 + 1))
    BCE = []
    rBCE = []
    for i in range(0, 5 + 1):
        if i == 0:
            BCE.append(csv_dict["s0cr0"].loc[0, 'mean'])
            rBCE.append(csv_dict["s0cr0"].loc[1, 'mean'])
        else:
            BCE.append(csv_dict[f"s{int(i)}cr{0.5}"].loc[0, 'mean'])
            rBCE.append(csv_dict[f"s{int(i)}cr{0.5}"].loc[1, 'mean'])

    BCE_clahe = []
    rBCE_clahe = []
    for i in range(0, 5 + 1):
        if i == 0:
            BCE_clahe.append(csv_dict_clahe["s0cr0"].loc[0, 'mean'])
            rBCE_clahe.append(csv_dict_clahe["s0cr0"].loc[1, 'mean'])
        else:
            BCE_clahe.append(csv_dict_clahe[f"s{int(i)}cr{0.5}"].loc[0, 'mean'])
            rBCE_clahe.append(csv_dict_clahe[f"s{int(i)}cr{0.5}"].loc[1, 'mean'])

    plt.plot(index, BCE, label="mBCE")
    plt.plot(index, BCE_clahe, label="mBCE (CLAHE)")
    plt.plot(index, rBCE, label="relative mBCE")
    plt.plot(index, rBCE_clahe, label="relative mBCE (CLAHE)")
    plt.legend()
    plt.xlabel("Corruption severity")
    plt.ylabel("(relative) mBCE")
    plt.title(f"(relative) mBCE for different corruption severity levels (cr = {0.5})")

    plt.show()

    os.chdir(base_dir)


def plot_BCE_cr(dir_path, clahe_dir_path = None):
    base_dir = os.getcwd()

    csv_dict = dict()
    os.chdir(dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" not in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    csv_dict_clahe = dict()
    os.chdir(base_dir)
    os.chdir(clahe_dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict_clahe["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict_clahe[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    index = np.array(range(int(0 * 10), int((0.5 + 0.1) * 10))) / 10
    BCE = []
    rBCE = []
    for i in range(int(0 * 10), int((0.5 + 0.1) * 10)):
        if i == 0:
            BCE.append(csv_dict["s0cr0"].loc[0, 'mean'])
            rBCE.append(csv_dict["s0cr0"].loc[1, 'mean'])
        else:
            BCE.append(csv_dict[f"s{3}cr{i / 10}"].loc[0, 'mean'])
            rBCE.append(csv_dict[f"s{3}cr{i / 10}"].loc[1, 'mean'])

    BCE_clahe = []
    rBCE_clahe = []
    for i in range(int(0 * 10), int((0.5 + 0.1) * 10)):
        if i == 0:
            BCE_clahe.append(csv_dict_clahe["s0cr0"].loc[0, 'mean'])
            rBCE_clahe.append(csv_dict_clahe["s0cr0"].loc[1, 'mean'])
        else:
            BCE_clahe.append(csv_dict_clahe[f"s{3}cr{i / 10}"].loc[0, 'mean'])
            rBCE_clahe.append(csv_dict_clahe[f"s{3}cr{i / 10}"].loc[1, 'mean'])

    plt.plot(index, BCE, label="mBCE")
    plt.plot(index, BCE_clahe, label="mBCE (CLAHE)")
    plt.plot(index, rBCE, label="relative mBCE")
    plt.plot(index, rBCE_clahe, label="relative mBCE (CLAHE)")
    plt.legend()
    plt.xlabel("Corruption ratio")
    plt.ylabel("(relative) mBCE")
    plt.title(f"(relative) mBCE for different corruption ratios (s = 3)")

    plt.show()

    os.chdir(base_dir)


def plot_clean(dir_path, clahe_dir_path):
    base_dir = os.getcwd()

    csv_dict = dict()
    os.chdir(dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" not in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    csv_dict_clahe = dict()
    os.chdir(base_dir)
    os.chdir(clahe_dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict_clahe["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict_clahe[
                    f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    index = list(range(0, 5 + 1))
    clean = []
    for i in range(0, 5 + 1):
        if i == 0:
            clean.append(csv_dict["s0cr0"].loc[0, 'clean error rate'])
        else:
            clean.append(csv_dict[f"s{int(i)}cr{0.5}"].loc[0, 'clean error rate'])

    clean_clahe = []
    for i in range(0, 5 + 1):
        if i == 0:
            clean_clahe.append(csv_dict_clahe["s0cr0"].loc[0, 'clean error rate'])
        else:
            clean_clahe.append(csv_dict_clahe[f"s{int(i)}cr{0.5}"].loc[0, 'clean error rate'])

    plt.plot(index, clean, label="Clean error rate")
    plt.plot(index, clean_clahe, label="Clean error rate (CLAHE)")
    plt.xlabel("Corruption severity")
    plt.ylabel("Clean error rate")
    plt.title(f"Clean error rate for different corruption severity levels (cr = {0.5})")
    plt.legend()

    plt.show()

    os.chdir(base_dir)


def plot_clean_cr(dir_path, clahe_dir_path):
    base_dir = os.getcwd()

    csv_dict = dict()
    os.chdir(dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" not in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    csv_dict_clahe = dict()
    os.chdir(base_dir)
    os.chdir(clahe_dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name and "clahe" in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict_clahe["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict_clahe[
                    f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    index = np.array(range(int(0 * 10), int((0.5 + 0.1) * 10))) / 10
    clean = []
    for i in range(int(0 * 10), int((0.5 + 0.1) * 10)):
        if i == 0:
            clean.append(csv_dict["s0cr0"].loc[0, 'clean error rate'])
        else:
            clean.append(csv_dict[f"s{3}cr{i / 10}"].loc[0, 'clean error rate'])

    clean_clahe = []
    for i in range(int(0 * 10), int((0.5 + 0.1) * 10)):
        if i == 0:
            clean_clahe.append(csv_dict_clahe["s0cr0"].loc[0, 'clean error rate'])
        else:
            clean_clahe.append(csv_dict_clahe[f"s{3}cr{i / 10}"].loc[0, 'clean error rate'])

    plt.plot(index, clean, label="Clean error rate")
    plt.plot(index, clean_clahe, label="Clean error rate (CLAHE)")
    plt.xlabel("Corruption ratio")
    plt.ylabel("Clean error rate")
    plt.title(f"Clean error rate for different corruption ratios (s = 3)")
    plt.legend()

    plt.show()

    os.chdir(base_dir)


plot_BCE(non_clahe_dir, clahe_dir)
plot_BCE_cr(non_clahe_dir, clahe_dir)
plot_clean(non_clahe_dir, clahe_dir)
plot_clean_cr(non_clahe_dir, clahe_dir)