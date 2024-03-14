import os
import keras
import numpy
import imageio

def load_batch(path, batch_size=1):
    if not os.path.exists(path):
        return [], []

    # Get all the color images inside path.
    batch_images = list(filter(os.path.isfile, os.listdir(path)))
    batch_images = [file.split("_")[1] for file in batch_images if file.startswith("color_")]

    if len(batch_images) > batch_size:
        batch_images = numpy.random.choice(batch_images, size=batch_size)

    originals = []
    grayscale = []
    for img_path in batch_images:
        color_img = imageio.imread(os.path.join(path, "color_"     + img_path)).astype(numpy.float)
        gray_img  = imageio.imread(os.path.join(path, "grayscale_" + img_path)).astype(numpy.float)
        originals.append(color_img)
        grayscale.append(gray_img)

    originals = numpy.array(originals) / 127.5 - 1.
    grayscale = numpy.array(grayscale) / 127.5 - 1.
    return originals, grayscale

def get_best_model(path):
    return keras.models.load_model(f"{path}/models/model_15_2500.h5")