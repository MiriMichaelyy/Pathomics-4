import os
# import glob
import keras
import numpy
import skimage
import imageio

def get_size(path, suffix="png"):
    return len(list(filter(lambda item: item.endswith(suffix), os.listdir(path))))

def load_dataset(path, suffix="png"):
    if not os.path.exists(path):
        return []

    dataset = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(suffix)]
    dataset = list(filter(os.path.isfile, dataset))
    dataset = list(map(os.path.abspath, dataset))

    for full_path in dataset:
        yield imageio.imread(full_path).astype(numpy.uint8)

def load_batch(path, size, suffix="png"):
    # paths = glob.glob(path + os.path.sep + "*." + suffix)
    paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(suffix)]
    paths = list(filter(os.path.isfile, paths))
    batch = numpy.random.choice(paths, size)
    for image in batch:
        image   = imageio.imread(image).astype(float)
        h, w, _ = image.shape
        half_w  = int(w / 2)

        color     = image[:, :half_w, :]
        grayscale = image[:, half_w:, :]

        color     = skimage.transform.resize(color,     (h, half_w), mode='reflect', anti_aliasing=True)
        grayscale = skimage.transform.resize(grayscale, (h, half_w), mode='reflect', anti_aliasing=True)

        if numpy.random.random() > 0.5:
            color     = numpy.fliplr(color)
            grayscale = numpy.fliplr(grayscale)

        color     = numpy.array([color])     / 127.5 - 1.
        grayscale = numpy.array([grayscale]) / 127.5 - 1.

        print(color.shape)
        yield color, grayscale

def get_best_model(path):
    loss_arr = []
    for path in glob.glob(os.path.join(path, "E(0-9)+_Disc_loss_real.npy")):
        loss_arr.append(numpy.load(path))

    best  = min(loss_arr)
    epoch = loss_arr.index(best)

    return keras.models.load_model(os.path.join(path, "models", f"g_model_{epoch}.keras"))
