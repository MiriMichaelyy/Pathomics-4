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

def split_dataset(color, grayscale, combined, split):
    num_of_samples = len(color)
    train_samples  = num_of_samples * split[0]
    test_samples   = num_of_samples * split[1]
    val_samples    = num_of_samples * split[2]

    indices = range(num_of_samples)
    random.shuffle(indices)

    train_indices = indices[:train_samples]
    test_indices  = indices[train_samples:train_samples + test_samples]
    val_indices   = indices[train_samples + test_samples:]

    # Create train, test, and val lists
    train = [(color[i], grayscale[i], combined[i]) for i in train_indices]
    test  = [(color[i], grayscale[i], combined[i]) for i in test_indices]
    val   = [(color[i], grayscale[i], combined[i]) for i in val_indices]

    return train, test, val

def preprocess(dataset, split, size):
    color_images     = []
    grayscale_images = []
    combined_images  = []
    for index, image in enumerate(dataset):
        color, grayscale, combined = process_image(iamge, size)
        color_images     += color
        combined_images  += combined
        grayscale_images += grayscale
    return split_dataset(color_images, grayscale_images, split)

def train(path, models, dataset, n_epochs=15, n_batch=1):
    Disc_loss_real = []
    Disc_loss_fake = []
    Gen_loss       = []
    d_model, g_model, gan_model = models

    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # calculate the number of training iterations
    # manually enumerate epochs
    y_real = numpy.ones((n_batch, n_patch, n_patch, 1))
    start_time = datetime.datetime.now()
    for i in range(n_epochs):

        # select a batch of real samples
        for batch_i, (X_realB, X_realA) in enumerate(dataset):

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

            # summarize model performance
            # set the number of times the model and images are saved
            if (batch_i + 1) % 500 == 0:
                outputs.summarize_performance(path, i, batch_i, g_model)

    # Save the loss values to NumPy files
    if not os.path.exists(f"{path}/results"):
        os.makedirs(f"{path}/results")
    numpy.save(f'{path}/results/Disc_loss_real.npy', numpy.array(Disc_loss_real))
    numpy.save(f'{path}/results/Disc_loss_fake.npy', numpy.array(Disc_loss_fake))
    numpy.save(f'{path}/results/Gen_loss.npy',       numpy.array(Gen_loss))

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

def test(model, dataset, img_shape):
    start_time = datetime.datetime.now()
    for index, (tar_image, src_image) in enumerate(dataset):
        print(f"Testing image #{index+1}")
        gen_image = model.predict(src_image)
        # plot_images(src_image, gen_image, tar_image, sample)
    print('time: ', datetime.datetime.now() - start_time)