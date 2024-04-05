import tifffile
import matplotlib.pyplot as plt
from skimage import exposure

# Load TIFF image
def load_tiff_image(img_path):
    return tifffile.imread(img_path)

# Plot intensity distribution and CDF on the same graph
def plot_intensity_distribution(image, ax, c_color, title):
    img_hist, bins = exposure.histogram(image, source_range='dtype')
    ax.plot(bins, img_hist / img_hist.max(), label='Intensity Histogram')
    img_cdf, bins = exposure.cumulative_distribution(image)
    ax.plot(bins, img_cdf, label='Cumulative Distribution Function')
    ax.set_ylabel(c_color)
    ax.legend()
    ax.set_title(title)

# Plot intensity distribution and CDF for source, reference, and normalized images
def plot_intensity_distribution_comparison(source_image, reference_image, normalized_image):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    for i, (image, title) in enumerate(zip([source_image, reference_image, normalized_image], ['Source', 'Reference', 'Normalized'])):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            plot_intensity_distribution(image[..., c], axes[c, i], c_color, title)

    # Remove empty plots
    for i in range(2, 3):
        for j in range(1, 3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# Paths to source, reference, and normalized images
source_img_path = r'C:\Users\mirim\PycharmProjects\STST_replication\New_data\results\Original\1.tiff'
reference_img_path = r'C:\Users\mirim\PycharmProjects\STST_replication\H\20.tiff'
normalized_img_path = r'C:\Users\mirim\PycharmProjects\STST_replication\New_data\results\Generated\1.tiff'

# Load the source, reference, and normalized images
source_image = load_tiff_image(source_img_path)
reference_image = load_tiff_image(reference_img_path)
normalized_image = load_tiff_image(normalized_img_path)

# Plot intensity distribution and CDF for source, reference, and normalized images
plot_intensity_distribution_comparison(source_image, reference_image, normalized_image)
