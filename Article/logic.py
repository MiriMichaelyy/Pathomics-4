import os
import numpy
import random
import imageio
import datetime

import inputs
import outputs

from PIL import Image

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
        combined = Image.new("RGB", (w*2, h))
        combined.paste(colored,   (0, 0, 1*w, h))
        combined.paste(grayscale, (w, 0, 2*w, h))
        combined_images.append(combined)

    return colored_images, grayscale_images, combined_images

def split_dataset(dataset, dataset_len, split):
    def gen_dataset(color, grayscale, combined, maximum):
        while maximum > 0:
            yield next(color), next(grayscale), next(combined)
            maximum -= 1

    color, grayscale, combined = dataset
    train_set = gen_dataset(color, grayscale, combined, dataset_len * split[0])
    test_set  = gen_dataset(color, grayscale, combined, dataset_len * split[1])
    val_set   = gen_dataset(color, grayscale, combined, dataset_len * split[2])

    return train_set, test_set, val_set

def preprocess(dataset, split, size):
    color_images     = []
    grayscale_images = []
    combined_images  = []

    for index, image in enumerate(dataset):
        color, grayscale, combined = process_image(image, size)
        color_images     += color
        combined_images  += combined
        grayscale_images += grayscale

    new_dataset = (color_images, grayscale_images, combined_images)
    return split_dataset(new_dataset, split)

def train(models, dataset):
    d_model, g_model, gan_model = models
    Disc_loss_real = []
    Disc_loss_fake = []
    Gen_loss       = []

    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # calculate the number of training iterations
    y_real = numpy.ones((1, n_patch, n_patch, 1))

    # select a batch of real samples
    start_time = datetime.datetime.now()
    for i, (X_realB, X_realA, combined) in enumerate(dataset):

        # Generate a batch of fake samples.
        X_fakeB = g_model.predict(X_realA)
        y_fake  = numpy.zeros((len(X_fakeB), n_patch, n_patch, 1))

        # Update models for real samples.
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        elapsed_time = datetime.datetime.now() - start_time

        # summarize performance
        # print('>step: %d >epoch %d-%d >batch %d-%d, D_loss_real[%.3f]  D_loss_fake[%.3f]  G_loss[%.3f]  time: %s' % ((batch_i + 1 + (i * bat_per_epo)), i + 1, n_epochs, batch_i + 1, bat_per_epo, d_loss1, d_loss2, g_loss, elapsed_time))

    # Save the loss values in the array
    Disc_loss_real.append(d_loss1)
    Disc_loss_fake.append(d_loss2)
    Gen_loss.append(g_loss)

    return (g_model, d_model, gan_model), (Disc_loss_real, Disc_loss_fake, Gen_loss)

# WHTA TO DO?
# # select a sample of input images
# [X_realB, X_realA] = inputs.load_batch(dataset, 3)
# # generate a batch of fake samples
# X_fakeB = g_model.predict(X_realA)

# # scale all pixels from [-1,1] to [0,1]
# X_realA = (X_realA + 1) / 2.0
# X_realB = (X_realB + 1) / 2.0
# X_fakeB = (X_fakeB + 1) / 2.0

# # for i in range(3):
#     # plt.imsave(f"{path}/results/Generated_B{batch+1}_{i}.tiff", X_fakeB[i])
#     # plt.imsave(os.path.join(path, "results", 'Generated_B%d_%d.tiff' % (batch + 1, i + 1), X_fakeB[i]))

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img, patche):
    src_im = numpy.squeeze(numpy.array(src_img))
    gen_im = numpy.squeeze(numpy.array(gen_img))
    tar_im = numpy.squeeze(numpy.array(tar_img))

    path_image = 'C:/Users/mirim/PycharmProjects/STST_replication/results/'

    imageio.imwrite(path_image + 'Input/%d.tiff' % (patche + 1), src_im)
    imageio.imwrite(path_image + 'Generated/%d.tiff' % (patche + 1), gen_im)
    imageio.imwrite(path_image + 'Original/%d.tiff' % (patche + 1), tar_im)
# __________________________________

def test(model, dataset):
    start_time = datetime.datetime.now()
    for index, (tar_image, src_image) in enumerate(dataset):
        print(f"Testing image #{index+1}")
        gen_image = model.predict(src_image)
        # plot_images(src_image, gen_image, tar_image, sample)
    print('time: ', datetime.datetime.now() - start_time)