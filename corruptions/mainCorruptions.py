from imagenet_c import corrupt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import random
import string


def all_corrupt_image(image_path, severity):
    corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                        'motion_blur', 'zoom_blur', 'brightness', 'brightness_down', 'contrast',
                        'black_corner', 'characters', 'elastic_transform', 'pixelate', 'jpeg_compression']
    imgs = []
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    imgs.append(('original', img_array))
    for corruption_type in corruption_types:
        if corruption_type == 'brightness_down':
            img_corrupt = brightness_down(img_array, severity)
        elif corruption_type == "black_corner":
            img_corrupt = black_corner(img_array, severity)
        elif corruption_type == "characters":
            img_corrupt = characters(img_array, severity)
        else:
            img_corrupt = corrupt(img_array, severity, corruption_type)
        imgs.append((corruption_type, img_corrupt))

    fig = plt.figure(figsize=(10, 7))
    rows = 3
    columns = 5

    for i in range(len(imgs)):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(imgs[i][1])
        plt.axis('off')
        if imgs[i][0] == "brightness":
            plt.title("Brightness Up")
        else:
            plt.title(string.capwords(imgs[i][0].replace("_", " ")))

    plt.show()


def brightness_down(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] - c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return (np.clip(x, 0, 1) * 255).astype(int)


def black_corner(x, severity=1):
    x_img = Image.fromarray(x)
    if severity == 1:
        corner_img = Image.open("blackCorner-1.png")
    elif severity == 2:
        corner_img = Image.open("blackCorner-2.png")
    elif severity == 3:
        corner_img = Image.open("blackCorner-3.png")
    elif severity == 4:
        corner_img = Image.open("blackCorner-4.png")
    elif severity == 5:
        corner_img = Image.open("blackCorner-5.png")

    x_img.paste(corner_img, (0, 0), mask=corner_img)

    x_array = np.array(x_img)

    return x_array

def characters(image_array, severity):
    img_height, img_width, _ = image_array.shape

    # Initialize a blank image with the same dimensions and convert to PIL Image
    pil_img = Image.fromarray(image_array)

    # Define a font
    font = ImageFont.load_default()

    # Define possible characters
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    # Define number of words to add based on severity
    max_words = severity

    # Create a drawing context
    draw = ImageDraw.Draw(pil_img)

    # Add random words
    for _ in range(max_words):
        # Choose a random word length
        word_length = random.randint(1, 10)
        # Generate a random word
        word = ''.join(random.choices(characters, k=word_length))
        # Choose a random position
        x = random.randint(0, img_width - 10*word_length)
        y = random.randint(0, img_height - 10)
        # Choose a random color
        color = tuple(np.random.randint(0, 256, size=3))
        # Draw the word
        draw.text((x, y), word, fill=color, font=font)

    # Convert back to numpy array
    result_array = np.array(pil_img)

    return result_array

all_corrupt_image("../archive/HAM10000_ordered_224_0.8_0.2/train+val/akiec/ISIC_0024329_224.jpg", 3)
