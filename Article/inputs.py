import os
import PIL
import glob
import keras
import numpy
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

def get_size(path, suffix="png"):
    return len(list(filter(lambda item: item.endswith(suffix), os.listdir(path))))

def load_images(path, suffix="png"):
    if not os.path.exists(path):
        return []

    dataset = [os.path.join(path, file) for file in os.listdir(path)]
    dataset = list(filter(os.path.isfile, dataset))
    dataset = list(filter(lambda item: item.endswith(suffix), dataset))
    dataset = list(map(os.path.abspath, dataset))

    for full_path in dataset:
        with PIL.Image.open(full_path) as image:
            yield image

def load_dataset(path, suffix="png"):
    if not os.path.exists(path):
        return []

    dataset = [os.path.join(path, file) for file in os.listdir(path)]
    dataset = list(filter(os.path.isfile, dataset))
    dataset = list(filter(lambda item: item.endswith(suffix), dataset))
    dataset = list(map(os.path.abspath, dataset))

    for full_path in dataset:
        yield imageio.imread(full_path).astype(numpy.uint8)

def load_batch(path, size, suffix="png"):
    paths = glob.glob(path + os.path.sep + "*." + suffix)
    n_batches = int(len(paths) / size)
    for i in range(n_batches):
        batch = paths[i * size:(i + 1) * size]
        imgs_A, imgs_B = [], []
        for img in batch:
            img = imageio.imread(img).astype(float)
            h, w, _ = img.shape
            half_w = int(w / 2)
            img_A = img[:, :half_w, :]
            img_B = img[:, half_w:, :]
            img_A = resize(img_A, (256,256), mode='reflect', anti_aliasing=True)
            img_B = resize(img_B, (256,256), mode='reflect', anti_aliasing=True)
            if numpy.random.random() > 0.5:
                img_A = numpy.fliplr(img_A)
                img_B = numpy.fliplr(img_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = numpy.array(imgs_A) / 127.5 - 1.
        imgs_B = numpy.array(imgs_B) / 127.5 - 1.
        yield imgs_A, imgs_B

def get_best_model(path):
    loss_arr = []
    for path in glob.glob(os.path.join(path, "E(0-9)+_Disc_loss_real.npy")):
        loss_arr.append(numpy.load(path))

    best  = min(loss_arr)
    epoch = loss_arr.index(best)

    return keras.models.load_model(os.path.join(path, "models", f"g_model_{epoch}.keras"))
