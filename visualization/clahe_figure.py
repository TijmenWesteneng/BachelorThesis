import matplotlib.pyplot as plt
import cv2

file_dirs = ["../archive/HAM10000_ordered_224_0.8_0.2/train+val/akiec/ISIC_0024329_224.jpg",
             "../archive/HAM10000_ordered_224_0.8_0.2/train+val/bcc/ISIC_0024331_224.jpg",
             "../archive/HAM10000_ordered_224_0.8_0.2/train+val/bkl/ISIC_0024312_224.jpg",
             "../archive/HAM10000_ordered_224_0.8_0.2/train+val/mel/ISIC_0024310_224.jpg"]

fig = plt.figure(0, (4, 8))

for i, file_dir in enumerate(file_dirs):

    img = cv2.imread(file_dir)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    lab_img[...,0] = clahe.apply(lab_img[...,0])

    img_bgr = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    fig.add_subplot(len(file_dirs), 2, i * 2 + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if i == 0:
        plt.title("Original Image")

    fig.add_subplot(len(file_dirs), 2, i * 2 + 2)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if i == 0:
        plt.title("CLAHE Image")

plt.show()