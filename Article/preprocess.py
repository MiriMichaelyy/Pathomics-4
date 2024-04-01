import PIL
import math

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

def split_dataset(datasets, dataset_length, split):
    def gen_dataset(dataset, maximum):
        for index, image in enumerate(dataset):
            if index < maximum:
                yield image
            else:
                return

    train_size = math.floor(dataset_length * split[0])
    test_size  = math.floor(dataset_length * split[1])
    val_size   = math.floor(dataset_length * split[2])

    if train_size + test_size + val_size < dataset_length:
        train_size += (dataset_length - train_size + test_size + val_size)

    train_set = tuple(map(lambda dataset: gen_dataset(dataset, train_size), datasets))
    test_set  = tuple(map(lambda dataset: gen_dataset(dataset, test_size),  datasets))
    val_set   = tuple(map(lambda dataset: gen_dataset(dataset, val_size),   datasets))

    return train_set, test_set, val_set