import os
import glob
import keras
import numpy
import imageio
from PIL import Image

def get_size(path):
    return len(os.listdir(path))

def load_dataset(path, suffix="png"):
    if not os.path.exists(path):
        return []

    dataset = [os.path.join(path, file) for file in os.listdir(path)]
    dataset = list(filter(os.path.isfile, dataset))
    dataset = list(filter(lambda item: item.endswith(suffix), dataset))
    dataset = list(map(os.path.abspath, dataset))

    for full_path in dataset:
        with Image.open(full_path) as image:
            yield image

def load_batch(datasets, size):
    for i, (color, grayscale) in enumerate(datasets):
        if i >= size:
            return
        color_arr = numpy.array(color).astype(float)     / 127.5 - 1.
        gray_arr  = numpy.array(grayscale).astype(float) / 127.5 - 1.
        yield color_arr, gray_arr

def get_best_model(path):
    loss_arr = []
    for path in glob.glob(os.path.join(path, "E(0-9)+_Disc_loss_real.npy")):
        loss_arr.append(numpy.load(path))

    best  = min(loss_arr)
    epoch = loss_arr.index(best)

    return keras.models.load_model(os.path.join(path, "models", f"g_model_{epoch}.keras"))