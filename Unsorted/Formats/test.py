import os
import PIL
import math
import numpy
import matplotlib
import matplotlib.pyplot as plt
import PIL.features
import imageio
import imageio.v2 as imageio

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.144])

color     = imageio.imread("1_arr.tiff")
grayscale = rgb2gray(color)
grayscale = numpy.round(grayscale).astype(numpy.uint8)
grayscale_as_rgb = numpy.stack([grayscale] * 3, axis=-1)

# plt.imsave("GRAYSCALE.tiff",     grayscale)
plt.imsave("GRAYSCALE_RGB.tiff", grayscale_as_rgb)

# full_path = "1.tiff"
# with PIL.Image.open(full_path) as image:
#     print(image.info)
#     image.save("BLACK_PIL.tiff")
    # a = numpy.array(image) / 255
    # plt.imsave("1_arr.tiff", a)
    # b = PIL.Image.fromarray((a*255).astype(numpy.uint8))
    # b.save("1_arr.tiff")

#     pil_size = os.path.getsize("BLACK_PIL.tiff")
#     plt_size = os.path.getsize("BLACK_PLT.tiff")

#     print("PIL: ", pil_size)
#     print("PLT: ", plt_size)

# with PIL.Image.open("ORG.tiff") as image:
#     print(image.info)
#     print(image.tag_v2)

# with PIL.Image.open("1.tiff") as image:
#     print(image.info)
#     print(image.tag_v2)

# with PIL.Image.open("BLACK.tiff") as image:
#     print(image.info)
#     print(image.tag_v2)

# with PIL.Image.open("BLACK_PIL.tiff") as image:
#     print(image.info)
#     print(image.tag_v2)

# with PIL.Image.open("BLACK_PLT.tiff") as image:
#     print(image.info)
#     print(image.tag_v2)

# with PIL.Image.open("BLACK_ARR.tiff") as image:
#     print(image.info)
#     print(image.tag_v2)