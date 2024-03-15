import os
import keras
import numpy
import imageio
from PIL import Image

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

def load_batch(datasets, size):
    color_arr = []
    gray_arr  = []
    for i, (color, grayscale) in enumerate(datasets):
        if i >= size:
            break
        color_arr.append(numpy.array(color).astype(float))
        gray_arr.append(numpy.array(grayscale).astype(float))

    color_arr = numpy.array(color_arr) / 127.5 - 1.
    gray_arr  = numpy.array(gray_arr)  / 127.5 - 1.

    return color_arr, gray_arr

def get_best_model(path):
    return keras.models.load_model(os.path.join(path, "models", "g_model_15.keras"))