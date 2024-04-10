import PIL
import math
import numpy

def create_patches(image, size):
    img_width, img_height, _ = image.shape
    width, height, _         = size
    for row in range(0, img_height, height):
        for col in range(0, img_width, width):
            if col + width > img_width or row + height > img_height:
                continue
            yield image[row : row + height, col : col + width, :]
            # yield image.crop((col, row, col + width, row + height))

def valid_image(image, white_pixel=192, black_pixel=64, valid_thershold=0.8):
    pixels = numpy.array(image, dtype=numpy.uint8)
    whites = 0
    blacks = 0

    for row in range(pixels.shape[0]):
        for col in range(pixels.shape[1]):
            # Check if the pixel is a black pixel.
            if numpy.all(pixels[row, col, :] < black_pixel):
                blacks += 1
            # Check if the pixel is a white pixel.
            if numpy.all(pixels[row, col, :] > white_pixel):
                whites += 1

    # Check if the image is mostly black or white.
    threshold = pixels.shape[0] * pixels.shape[1] * valid_thershold
    if whites > threshold or blacks > threshold:
        return False
    return True

def process_image(image, size, no_filter=False):
    color_images     = []
    grayscale_images = []
    combined_images  = []

    # Convert Numpy array to Pillow image.
    for index, color in enumerate(create_patches(image, size)):
        if not no_filter and not valid_image(color):
            print(f"Patch #{index} is invalid")
            continue

        # Create grayscale image.
        PILImage  = PIL.Image.fromarray(color)
        PILgrey   = PILImage.convert('LA')
        grayscale = numpy.pad(numpy.array(PILgrey), ((0,0),(0,0),(0,1)), mode='constant', constant_values=0)

        # Create combined image.
        combined = numpy.hstack((color, grayscale))
        # w, h     = *size
        # combined = PIL.Image.new("RGB", (w*2, h))
        # combined.paste(color,     (0, 0, 1*w, h))
        # combined.paste(grayscale, (w, 0, 2*w, h))

        # Convert Pillow image to Numpy array.
        color_images.append(color)
        grayscale_images.append(grayscale)
        combined_images.append(combined)
        # color_images.append(numpy.array(color,         dtype=float) / 255)
        # grayscale_images.append(numpy.array(grayscale, dtype=float) / 255)
        # combined_images.append(numpy.array(combined,   dtype=float) / 255)

    return color_images, grayscale_images, combined_images

def split_dataset(datasets, dataset_length, split):
    def gen_dataset(dataset, maximum):
        for index, image in enumerate(dataset):
            if index < maximum:
                yield image
            else:
                return

    train_size  = math.floor(dataset_length * split[0])
    test_size   = math.floor(dataset_length * split[1])
    val_size    = math.floor(dataset_length * split[2])
    train_size += dataset_length - (train_size + test_size + val_size)

    train_set = tuple(map(lambda dataset: gen_dataset(dataset, train_size), datasets))
    test_set  = tuple(map(lambda dataset: gen_dataset(dataset, test_size),  datasets))
    val_set   = tuple(map(lambda dataset: gen_dataset(dataset, val_size),   datasets))

    return train_set, test_set, val_set
