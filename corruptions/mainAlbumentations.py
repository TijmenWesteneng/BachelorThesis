import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

transform = A.Compose([
    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    #A.Transpose(p=0.5)
])

image = Image.open("../HAM10000_ordered/akiec/ISIC_0024329.jpg")
image_array = np.array(image)

transformed = transform(image=image_array)
transformed_image = transformed["image"]

plt.imshow(transformed_image)
plt.show()