import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import string
import numpy as np

parser = argparse.ArgumentParser(
    prog="mean_comparison",
    description="Visualize different (relative) mBCE's in matplotlib charts"
)

parser.add_argument('dir_path', nargs="?", default="../remote_development/tests/20_epochs_3_earlystopping_32_batch/BCE",
                    help="Directory where BCE files are located", type=str)
parser.add_argument("-s", "--severity",
                    help="Plot (relative) mBCE over different severities", action="store_true")
parser.add_argument("-c", "--corruption",
                    help="Plot (relative) mBCE over different corruption ratios", action="store_true")
parser.add_argument( "--clean",
                    help="Plot clean error rate", action="store_true")
parser.add_argument("--s_min", default=0, type=float,
                    help="Minimum severity / corruption ratio")
parser.add_argument("--s_max", default=5, type=float,
                    help="Maximum severity / corruption ratio")
parser.add_argument("--cr", default=0.5, type=float,
                    help="Maximum severity / corruption ratio")

args = parser.parse_args()


def plot_BCE(corruption, dir_path, s_min = 0, s_max = 5, cr = 0.5, clahe = False):
    csv_dict = dict()
    os.chdir(dir_path)
    for file_name in os.listdir():
        if ".csv" in file_name:
            s_pos = file_name.find("_s")
            cr_pos = file_name.find("_cr")
            if s_pos == -1 and cr_pos == -1:
                csv_dict["s0cr0"] = pd.read_csv(file_name)
            else:
                csv_dict[f"s{file_name[s_pos + 2: s_pos + 3]}cr{file_name[cr_pos + 3: cr_pos + 6]}"] = pd.read_csv(file_name)

    if not corruption:
        if not args.clean:
            index = list(range(s_min, s_max + 1))
            BCE = []
            rBCE = []
            for i in range(s_min, s_max + 1):
                if i == 0:
                    BCE.append(csv_dict["s0cr0"].loc[0, 'mean'])
                    rBCE.append(csv_dict["s0cr0"].loc[1, 'mean'])
                else:
                    BCE.append(csv_dict[f"s{int(i)}cr{cr}"].loc[0, 'mean'])
                    rBCE.append(csv_dict[f"s{int(i)}cr{cr}"].loc[1, 'mean'])

            plt.plot(index, BCE, label="mBCE")
            plt.plot(index, rBCE, label="relative mBCE")
            plt.legend()
            plt.xlabel("Corruption severity")
            plt.ylabel("(relative) mBCE")
            plt.title(f"(relative) mBCE for different corruption severity levels (cr = {cr})")
        else:
            index = list(range(int(s_min), int(s_max + 1)))
            clean = []
            for i in range(int(s_min), int(s_max + 1)):
                if i == 0:
                    clean.append(csv_dict["s0cr0"].loc[0, 'clean error rate'])
                else:
                    clean.append(csv_dict[f"s{int(i)}cr{cr}"].loc[0, 'clean error rate'])

            plt.plot(index, clean, label="Clean error rate")
            plt.xlabel("Corruption severity")
            plt.ylabel("Clean error rate")
            plt.title(f"Clean error rate for different corruption severity levels (cr = {cr})")

    # Plot varied corruption ratio, keep corruption severity fixed
    else:
        if not args.clean:
            index = np.array(range(int(s_min * 10), int((s_max + 0.1) * 10))) / 10
            BCE = []
            rBCE = []
            for i in range(int(s_min * 10), int((s_max + 0.1) * 10)):
                if i == 0:
                    BCE.append(csv_dict["s0cr0"].loc[0, 'mean'])
                    rBCE.append(csv_dict["s0cr0"].loc[1, 'mean'])
                else:
                    BCE.append(csv_dict[f"s{int(cr)}cr{i / 10}"].loc[0, 'mean'])
                    rBCE.append(csv_dict[f"s{int(cr)}cr{i / 10}"].loc[1, 'mean'])

            plt.plot(index, BCE, label="mBCE")
            plt.plot(index, rBCE, label="relative mBCE")
            plt.legend()
            plt.xlabel("Corruption ratio")
            plt.ylabel("(relative) mBCE")
            plt.title(f"(relative) mBCE for different corruption ratios (s = {int(cr)})")
        else:
            index = np.array(range(int(s_min * 10), int((s_max + 0.1) * 10))) / 10
            clean = []
            for i in range(int(s_min * 10), int((s_max + 0.1) * 10)):
                if i == 0:
                    clean.append(csv_dict["s0cr0"].loc[0, 'clean error rate'])
                else:
                    clean.append(csv_dict[f"s{int(cr)}cr{i / 10}"].loc[0, 'clean error rate'])

            plt.plot(index, clean, label="Clean error rate")
            plt.xlabel("Corruption ratio")
            plt.ylabel("Clean error rate")
            plt.title(f"Clean error rate for different corruption ratios (s = {int(cr)})")

    plt.show()


if __name__ == "__main__":
    plot_BCE(args.corruption, args.dir_path, args.s_min, args.s_max, args.cr)