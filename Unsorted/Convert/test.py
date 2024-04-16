import os
import PIL
import numpy
import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt

def print_diff(in_image, name, extn, frmt):
    in_size = os.path.getsize(in_image)
    out_size = os.path.getsize(f"{extn}\\{name}.{frmt}")

    if in_size != out_size:
        diff = in_size - out_size
        unit = "B"
        if abs(diff) > 1073741824:
            diff /= 1073741824
            unit = "GB"
        elif abs(diff) > 1048576:
            diff /= 1048576
            unit = "MB"
        elif abs(diff) > 1024:
            diff /= 1024
            unit = "KB"

        print(f"Diff {extn}->{frmt} of %.3f {unit}." % diff)


def convert_pillow(in_image, formats):
    name = in_image.split('.')[0]
    extn = in_image.split('.')[1]

    with PIL.Image.open(in_image) as image:
        for frmt in formats:
            image.save(f"{extn}\\{name}.{frmt}")
            print_diff(in_image, name, extn, frmt)

def convert_imageio_uint(in_image, formats):
    name = in_image.split('.')[0]
    extn = in_image.split('.')[1]

    byte_array = imageio.imread(in_image).astype(numpy.uint8)
    for frmt in formats:
        plt.imsave(f"{extn}\\{name}.{frmt}", byte_array)
        print_diff(in_image, name, extn, frmt)

def convert_imageio_float(in_image, formats):
    name = in_image.split('.')[0]
    extn = in_image.split('.')[1]

    byte_array = imageio.imread(in_image).astype(float) / 255
    for frmt in formats:
        plt.imsave(f"{extn}\\{name}.{frmt}", byte_array)
        print_diff(in_image, name, extn, frmt)

def convert_imageio_pil(in_image, formats):
    name = in_image.split('.')[0]
    extn = in_image.split('.')[1]

    byte_array = imageio.imread(in_image).astype(numpy.uint8)
    image      = PIL.Image.fromarray(byte_array)
    byte_array = numpy.array(image)
    for frmt in formats:
        plt.imsave(f"{extn}\\{name}.{frmt}", byte_array)
        print_diff(in_image, name, extn, frmt)

print("PILLOW convert tests [png, jpg, tiff, scn] -> [png, jpg, tiff]")
convert_pillow("test.png",  ["png", "jpg", "tiff"])
convert_pillow("test.jpg",  ["png", "jpg", "tiff"])
convert_pillow("test.tiff", ["png", "jpg", "tiff"])
convert_pillow("test.scn",  ["png", "jpg", "tiff"])

print("ImageIO uint convert tests [png, jpg, tiff, scn] -> [png, jpg, tiff]")
convert_imageio_uint("test.png",  ["png", "jpg", "tiff"])
convert_imageio_uint("test.jpg",  ["png", "jpg", "tiff"])
convert_imageio_uint("test.tiff", ["png", "jpg", "tiff"])
convert_imageio_uint("test.scn",  ["png", "jpg", "tiff"])

print("ImageIO float convert tests [png, jpg, tiff, scn] -> [png, jpg, tiff]")
convert_imageio_float("test.png",  ["png", "jpg", "tiff"])
convert_imageio_float("test.jpg",  ["png", "jpg", "tiff"])
convert_imageio_float("test.tiff", ["png", "jpg", "tiff"])
convert_imageio_float("test.scn",  ["png", "jpg", "tiff"])

print("ImageIO PIL convert tests [png, jpg, tiff, scn] -> [png, jpg, tiff]")
convert_imageio_pil("test.png",  ["png", "jpg", "tiff"])
convert_imageio_pil("test.jpg",  ["png", "jpg", "tiff"])
convert_imageio_pil("test.tiff", ["png", "jpg", "tiff"])
convert_imageio_pil("test.scn",  ["png", "jpg", "tiff"])