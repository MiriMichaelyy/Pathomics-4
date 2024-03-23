import os
import glob
import keras
import numpy
import imageio
from PIL import Image

def get_size(path):
    return len(os.listdir(path))

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
    for i, (color, grayscale) in enumerate(datasets):
        if i >= size:
            return
        color_arr = numpy.array(color).astype(float)     / 127.5 - 1.
        gray_arr  = numpy.array(grayscale).astype(float) / 127.5 - 1.
        yield color_arr, gray_arr

def get_best_model(path):
    paths = glob.glob("E(0-9)+_Disc_loss_real.npy")
    loss_disc_real = []
    for path in enumerate(paths):
        numpy.load(path)

    losses = numpy.load('Disc_loss_fake.npy')
    loss_disc_real = numpy.save(os.path.join(path, f'E{epoch}_Disc_loss_real.npy'), numpy.array(losses[0]))
    loss_disc_fake = numpy.save(os.path.join(path, f'E{epoch}_Disc_loss_fake.npy'), numpy.array(losses[1]))
    loss_gen       = numpy.save(os.path.join(path, f'E{epoch}_Gen_loss.npy'),       numpy.array(losses[2]))

    return keras.models.load_model(os.path.join(path, "models", "g_model_15.keras"))