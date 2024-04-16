import os
import PIL
import glob
import keras
import numpy
import skimage
import imageio

def get_size(path, suffix="png"):
    return len(list(filter(lambda item: item.endswith(suffix), os.listdir(path))))

def convert(image):
    return numpy.array([image]) / 127.5 - 1.

def load_dataset(path, suffix="png"):
    if not os.path.exists(path):
        return []

    dataset = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(suffix)]
    dataset = list(filter(os.path.isfile, dataset))
    dataset = list(map(os.path.abspath, dataset))

    for full_path in dataset:
        image = imageio.imread(full_path)
        if image.shape[2] == 4:
            image = image[:,:,:3]
        yield image

def load_batch(path, size, suffix="png"):
    # paths = glob.glob(path + os.path.sep + "*." + suffix)
    paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(suffix)]
    paths = list(filter(os.path.isfile, paths))
    batch = numpy.random.choice(paths, size, False)
    for image in batch:
        color_arr     = []
        grayscale_arr = []

        image   = imageio.imread(image).astype(float)
        h, w, _ = image.shape
        half_w  = int(w / 2)

        color     = image[:, :half_w, :3]
        grayscale = image[:, half_w:, :3]

        color     = skimage.transform.resize(color,     (h, half_w), mode='reflect', anti_aliasing=True)
        grayscale = skimage.transform.resize(grayscale, (h, half_w), mode='reflect', anti_aliasing=True)

        if numpy.random.random() > 0.5:
            color     = numpy.fliplr(color)
            grayscale = numpy.fliplr(grayscale)

        yield convert(color), convert(grayscale)

def get_best_model(path):
    min_epoch = 1
    min_loss  = numpy.load(os.path.join(path, "E1_Gen_loss.npy"))

    for array in glob.glob(os.path.join(path, "E*_Gen_loss.npy")):
        epoch = int(array.split('E')[1].split('_')[0])
        loss  = numpy.load(array)

        if numpy.sum(loss) < numpy.sum(min_loss):
            min_epoch = epoch
            min_loss  = loss

    model = keras.models.load_model(os.path.join(path, "models", f"g_model_{min_epoch}.keras"))
    return model, min_epoch