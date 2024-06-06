import os
import pandas as pd


dirs = [
    "tests/20_epochs_3_earlystopping_64_batch_2/non-clahe (trash - wrong test set)/BCE",
    "tests/20_epochs_3_earlystopping/BCE",
        ]

save_dir = "tests/mean_20_epochs_3_earlystopping_32_batch"


def list_and_average(directories, csv_save_dir):
    csv_list = os.listdir(directories[0])

    for csv_file in csv_list:
        dataframes = []

        for dir in directories:
            if not os.path.isdir(dir):
                print(f"WARNING: {dir} doesn't exist")
                continue

            csv_path = os.path.join(dir, csv_file)

            if not os.path.isfile(csv_path):
                print(f"WARNING: {csv_path} doesn't exist")
            else:
                df = pd.read_csv(csv_path, index_col=0)
                dataframes.append(df)

        mean_df = average_dataframes(dataframes)
        print(f"Averaged {csv_file} ({len(dataframes)} files)")

        csv_save_path = os.path.join(csv_save_dir, f"mean_{csv_file}")
        mean_df.to_csv(csv_save_path)
        print(f"Saved mean to {csv_save_path}")


def average_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    # Ensure all DataFrames have the same shape and align their indices and columns
    for df in dataframes:
        assert dataframes[0].shape == df.shape, "DataFrames do not have the same shape"
        assert (dataframes[0].columns == df.columns).all(), "DataFrame columns do not match"
        assert (dataframes[0].index == df.index).all(), "DataFrame indices do not match"

    # Concatenate DataFrames along a new dimension (run)
    concat_df = pd.concat(dataframes, keys=range(len(dataframes)), names=['run'])

    # Calculate the mean across the 'run' level
    mean_df = concat_df.groupby(level=-1).mean().round(3)

    return mean_df


if __name__ == "__main__":
    list_and_average(dirs, save_dir)