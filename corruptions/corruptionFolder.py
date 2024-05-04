import os
from pathlib import Path
from imagenet_c import corrupt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import skimage as sk
import random
from tqdm.auto import tqdm

src_dir = "../archive/HAM10000_augmented_all224"
des_dir = "../HAM10000_augmented_all224_corrupted"


def corrupt_folder(src, des):
    corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                        'motion_blur', 'zoom_blur', 'brightness', 'brightness_down', 'contrast',
                        'black_corner', 'characters', 'elastic_transform', 'pixelate', 'jpeg_compression']

    # Loop over all the files in the source directory
    for file in tqdm(os.listdir(src)):
        # If file was already corrupted, continue
        if "5" in os.listdir(des) and Path(file).stem + "_5_jpeg_compression.jpg" in os.listdir(os.path.join(des, "5", "jpeg_compression")):
            continue

        img_src = os.path.join(src, file)
        img = Image.open(img_src)
        img_array = np.array(img)

        # Loop over severity 1 to 5
        for severity in range(1, 6):
            # Loop over all corruption types and corrupt the images accordingly
            for corruption_type in corruption_types:
                if corruption_type == 'brightness_down':
                    img_corrupt = brightness_down(img_array, severity)
                elif corruption_type == "black_corner":
                    img_corrupt = black_corner(img_array, severity)
                elif corruption_type == "characters":
                    img_corrupt = characters(img_array, severity)
                else:
                    img_corrupt = corrupt(img_array, severity, corruption_type)

                # Convert image numpy array back to PIL image
                res_img = Image.fromarray(img_corrupt)

                # If severity dir doesn't exist yet, create it
                sev_dir = os.path.join(des, str(severity))
                if str(severity) not in os.listdir(des):
                    os.mkdir(sev_dir)

                # If severity corruption dir doesn't exist yet, create it
                sev_cor_dir = os.path.join(sev_dir, corruption_type)
                if corruption_type not in os.listdir(sev_dir):
                    os.mkdir(sev_cor_dir)

                # Remove ext (.jpg) from file name and create destination path
                file_no_ext = Path(file).stem
                des_path = os.path.join(sev_cor_dir, file_no_ext + "_" + str(severity) + "_" + corruption_type + ".jpg")

                # Save the image in the destination directory
                res_img.save(des_path)


def corrupt_folders(src, des):
    # Loop over all folders within the folder and run corrupt_folder on them
    for folder in tqdm(os.listdir(src)):
        src_path = os.path.join(src, folder)
        des_path = os.path.join(des, folder)
        if folder not in os.listdir(des):
            os.mkdir(des_path)
        corrupt_folder(src_path, des_path)


def brightness_down(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] - c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


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


corrupt_folders(src_dir, des_dir)
