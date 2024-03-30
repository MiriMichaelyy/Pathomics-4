import os
import sys
from keras.models import load_model
import scipy.misc
import numpy as np
import datetime
import imageio
from skimage.transform import resize
import PIL
import math
from PIL import Image
import os

if len(sys.argv) <= 1:
    print("Missing dataset argument.")
    exit()

dataset_path = os.path.abspath(sys.argv[1])
if not os.path.exists(dataset_path):
    print(f"The dataset path ({dataset_path}) does not exists.")
    exit()


##############################
# PARAMETERS                 #
##############################
channels      = 3
img_rows      = 256
img_cols      = 256
img_shape     = (img_rows, img_cols, channels)
img_res = (img_rows, img_cols)



# Create directories.
processed_path = os.path.join(dataset_path, "Processed")
os.makedirs(processed_path,                                     exist_ok=True)
os.makedirs(os.path.join(processed_path, "color"),              exist_ok=True)
os.makedirs(os.path.join(processed_path, "grayscale"),          exist_ok=True)
os.makedirs(os.path.join(processed_path, "combined"),           exist_ok=True)

# Create results directory
results_path = os.path.join(dataset_path, "Results")
os.makedirs(results_path,                                     exist_ok=True)

# Path where model is saved
#model_path = ''



#############################################################################################
# Image preprocess - create&save color, grayscale (gs) and combined patches:


# Define preprocessing fn's
def load_dataset(path):
    if not os.path.exists(path):
        return []

    # dataset = list(map(lambda file: os.path.join(path, file), os.listdir(path)))
    dataset = [os.path.join(path, file) for file in os.listdir(path)]
    dataset = list(filter(os.path.isfile, dataset))
    dataset = list(filter(lambda item: item.endswith(".tiff"), dataset))
    dataset = list(map(lambda item: os.path.join(path, item), dataset))

    for full_path in dataset:
        with Image.open(full_path) as image:
            yield image
def create_patches(image, size):
    img_width, img_height   = image.size
    width, height, channels = size
    for row in range(0, img_height, height):
        for col in range(0, img_width, width):
            if col + width > img_width or row + height > img_height:
                continue
            yield image.crop((col, row, col + width, row + height))

def process_image(image, size):
    colored_images   = []
    combined_images  = []
    grayscale_images = []

    for index, colored in enumerate(create_patches(image, size)):
        colored_images.append(colored)
        grayscale = colored.convert('LA')
        grayscale_images.append(grayscale)

        w, h     = colored.size
        combined = PIL.Image.new("RGB", (w*2, h))
        combined.paste(colored,   (0, 0, 1*w, h))
        combined.paste(grayscale, (w, 0, 2*w, h))
        combined_images.append(combined)

    return colored_images, grayscale_images, combined_images

def save_original(path, dataset, offset):
    os.makedirs(path, exist_ok=True)
    for index, image in enumerate(dataset):
        image.save(os.path.join(path, f"{offset + index + 1}.tiff"))
    return index + 1

def save_originals(path, color, grayscale, combined, total):
    count = save_original(os.path.join(path, "color"),     color,     total)
    count = save_original(os.path.join(path, "grayscale"), grayscale, total)
    count = save_original(os.path.join(path, "combined"),  combined,  total)
    return count



# Run preprocessing

total = 0
print(f"Splitting original images into {img_cols}x{img_rows} (color, grayscale & combined).")
for image in load_dataset(dataset_path):
    color, grayscale, combined = process_image(image, img_shape)
    total += save_originals(processed_path, color, grayscale, combined, total)

# Load dataset generators.
print("Loading color & grayscale datasets.")
color     = load_dataset(os.path.join(processed_path, "color"))
grayscale = load_dataset(os.path.join(processed_path, "grayscale"))


###########################################################################################
total_patches = total


# Define fn's for running the model

# Load preprocessed image patches
def load_data(sample_g):
    imgs_A = []
    imgs_B = []

    # paired dataset path
    path = processed_path/combined

    img = imageio.imread(path + '%d.tiff' % (sample_g + 1)).astype(np.float)
    h, w, _ = img.shape
    _w = int(w / 2)
    img_A, img_B = img[:, :_w, :], img[:, _w:, :]

    img_A = resize(img_A, img_res)
    img_B = resize(img_B, img_res)

    imgs_A.append(img_A)
    imgs_B.append(img_B)
    imgs_A = np.array(imgs_A) / 127.5 - 1.
    imgs_B = np.array(imgs_B) / 127.5 - 1.

    return imgs_A, imgs_B


# Plot source(input = gs), generated and target(original = color) images
def plot_images(gs_patch, gen_patch, orig_patch, patch):
    gs_im = np.squeeze(np.array(gs_patch))
    gen_im = np.squeeze(np.array(gen_patch))
    orig_im = np.squeeze(np.array(orig_patch))

    path_image = results_path


    imageio.imwrite(path_image + 'Input/%d.tiff' % (patch + 1), gs_im)
    imageio.imwrite(path_image + 'Generated/%d.tiff' % (patch + 1), gen_im)
    imageio.imwrite(path_image + 'Original/%d.tiff' % (patch + 1), orig_im)
# __________________________________

# Run normalization


# Load model
model = load_model(model_path)


start_time = datetime.datetime.now()
for patch in range(total_patches):
    # Load dataset
    [orig_patch, gs_patch] = load_data(patch)

    # Generate normalized batch
    gen_patch = model.predict(gs_patch)
    print('Generating normalized patch', patch + 1)

    # Plot all three images
    plot_images(gs_patch, gen_patch, orig_patch, patch)

elapsed_time = datetime.datetime.now() - start_time
print('time: ', elapsed_time)