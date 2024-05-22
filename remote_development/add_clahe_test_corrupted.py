import os

from dataset_management.add_clahe import add_clahe_to_dataset


src_folder = "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted/test"
des_folder = "../archive/test/HAM10000_ordered_224_0.8_0.2_corrupted_clahe/test"

# Loop over all severities, then over all corruptions and call the add_clahe function on each folder
for severity in os.listdir(src_folder):
    src_severity = os.path.join(src_folder, severity)
    for corruption in os.listdir(src_severity):
        src_severity_corruption = os.path.join(src_severity, corruption)
        des_severity_corruption = os.path.join(des_folder, severity, corruption)
        add_clahe_to_dataset(src_severity_corruption, des_severity_corruption)