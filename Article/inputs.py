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
    dataset = list(filter(lambda image: image.endswith(".tiff"), dataset))

    for full_path in dataset:
        with Image.open(full_path) as image:
            yield image

def load_batch(dataset, batch_size=1):
    random.shuffle(dataset)
    batch   = dataset[:batch_size]
    dataset = dataset[batch_size:]

    originals = []
    grayscale = []
    for colored, grayscale, combined in batch:
        originals.append(numpy.array(colored).astype(numpy.float))
        grayscale.append(numpy.array(grayscale).astype(numpy.float))

    originals = numpy.array(originals) / 127.5 - 1.
    grayscale = numpy.array(grayscale) / 127.5 - 1.

    return originals, grayscale

def get_best_model(path):
    return keras.models.load_model(f"{path}/models/model_15.h5")