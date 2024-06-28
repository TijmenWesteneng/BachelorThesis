import os
import matplotlib.pyplot as plt

dir = "../archive/HAM10000_ordered_224_0.8_0.2/train+val"

figure = plt.figure(0, (7, 7))

for i, folder in enumerate(os.listdir(dir)):
    dir_folder = os.path.join(dir, folder)
    for filename in os.listdir(dir_folder):
        file_path = os.path.join(dir_folder, filename)
        img = plt.imread(file_path)

        figure.add_subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(folder, fontsize="xx-large")
        plt.axis('off')

        break

plt.show()