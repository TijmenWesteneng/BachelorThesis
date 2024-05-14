from dataset_management.add_corruptions_to_dataset import add_corruptions_to_dataset
from augmentations.mainAugmentations import augment_folders

"""
for sev in range(1, 6):
    cor_rat = 0.5
    print(f"Adding corruptions with severity: {sev} and cor_ratio: {cor_rat}")
    add_corruptions_to_dataset("../archive/HAM10000_ordered_224_0.8_0.2/train+val",
                               "../archive/HAM10000_ordered_224_0.8_0.2_corrupted/train+val",
                               f"../archive/HAM10000_ordered_224_0.8_0.2_corrupted_s{sev}_cr{cor_rat}/train+val",
                               cor_ratio=cor_rat, sev=sev)
    print(f"Augmenting severity: {sev} and cor_ratio: {cor_rat}")
    augment_folders(f"../archive/HAM10000_ordered_224_0.8_0.2_corrupted_s{sev}_cr{cor_rat}/train+val",
                    f"../archive/train+val/HAM10000_ordered_224_0.8_0.2_corrupted_s{sev}_cr{cor_rat}_augmented/train+val",
                    ["nv"])
"""

for cor_ratio in range (1, 5):
    cor_rat = cor_ratio / 10
    sev = 3
    add_corruptions_to_dataset("../archive/HAM10000_ordered_224_0.8_0.2/train+val",
                               "../archive/HAM10000_ordered_224_0.8_0.2_corrupted/train+val",
                               f"../archive/HAM10000_ordered_224_0.8_0.2_corrupted_s{sev}_cr{cor_rat}/train+val",
                               cor_ratio=cor_rat, sev=sev)
    augment_folders(f"../archive/HAM10000_ordered_224_0.8_0.2_corrupted_s{sev}_cr{cor_rat}/train+val",
                    f"../archive/train+val/HAM10000_ordered_224_0.8_0.2_corrupted_s{sev}_cr{cor_rat}_augmented/train+val",
                    ["nv"])
