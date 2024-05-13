#!/bin/python3
import pathlib
import argparse

import imageio
import skimage
import matplotlib.pyplot as plt

##############################
# ARGUMENTS & PARAMETERS     #
##############################
parser = argparse.ArgumentParser(description="Display the intensity distribution of images.")
parser.add_argument("--images", required=True, nargs='+', default=[], help="List of images to analyze.")
args = parser.parse_args()

# Plot intensity distribution and CDF on the same graph
def plot_intensity_distribution(image, ax, c_color, title):
    img_hist, bins = skimage.exposure.histogram(image, source_range='dtype')
    ax.plot(bins, img_hist / img_hist.max(), label='Intensity Histogram')
    img_cdf, bins = skimage.exposure.cumulative_distribution(image)
    ax.plot(bins, img_cdf, label='Cumulative Distribution Function')
    ax.set_ylabel(c_color)
    ax.legend()
    ax.set_title(title)

names  = []
images = []
for path in args.images:
    images.append(imageio.imread(path))
    names.append(pathlib.Path(path).stem.title())

# Plot intensity distribution and CDF for source, and normalized images
fig, axes = plt.subplots(nrows=3, ncols=len(images), figsize=(10, 10))

for i, (image, title) in enumerate(zip(images, names)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        plot_intensity_distribution(image[..., c], axes[c, i], c_color, title)

plt.tight_layout()
plt.show()