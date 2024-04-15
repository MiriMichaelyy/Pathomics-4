import os
import numpy
import imageio
import matplotlib
import matplotlib.pyplot as plt

def convert(image):
    return numpy.squeeze(numpy.array(image))

def save_image(path, image, offset, suffix="png"):
    os.makedirs(path, exist_ok=True)
    imageio.imwrite(os.path.join(path, f"{offset}.{suffix}"), image)

def save_dataset(path, dataset, offset=0, suffix="png"):
    os.makedirs(path, exist_ok=True)
    count = 0
    for image in dataset:
        count += 1
        rgb_image = numpy.array(image) / 255
        plt.imsave(os.path.join(path, f"{offset + count}.{suffix}"), rgb_image)
    return count

def save_losses(path, losses, epoch):
    numpy.save(os.path.join(path, f'E{epoch}_Disc_real_loss.npy'), numpy.array(losses[0]))
    numpy.save(os.path.join(path, f'E{epoch}_Disc_fake_loss.npy'), numpy.array(losses[1]))
    numpy.save(os.path.join(path, f'E{epoch}_Gen_loss.npy'),       numpy.array(losses[2]))

def save_models(path, models, epoch):
    d_model, g_model, gan_model = models
    d_model.save  (os.path.join(path, "models", f"d_model_{epoch}.keras"))
    g_model.save  (os.path.join(path, "models", f"g_model_{epoch}.keras"))
    gan_model.save(os.path.join(path, "models", f"gan_model_{epoch}.keras"))

def smooth_curve_gen(points, factor=0.6):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def smooth_curve_dis(points, factor=0.3):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_outputs(path, epoch, size):
    Gen_loss       = numpy.load(os.path.join(path, f"E{epoch}_Gen_loss.npy"))
    Disc_real_loss = numpy.load(os.path.join(path, f"E{epoch}_Disc_real_loss.npy"))
    Disc_fake_loss = numpy.load(os.path.join(path, f"E{epoch}_Disc_fake_loss.npy"))
    
    temp = []
    for index, value in enumerate(Gen_loss):
        if (index + 1) % size:
            temp.append(value)
    Gen_loss = numpy.array(temp)
    temp = []
    for index, value in enumerate(Disc_real_loss):
        if (index + 1) % size:
            temp.append(value)
    Disc_real_loss = numpy.array(temp)
    temp = []
    for index, value in enumerate(Disc_fake_loss):
        if (index + 1) % size:
            temp.append(value)
    Disc_fake_loss = numpy.array(temp)

    epoch = range(0, len(Disc_real_loss))
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(epoch, Disc_real_loss, linewidth=0.8, color='#9b0000', marker='.')
    plt.title('Discriminator loss real')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(epoch, smooth_curve_dis(Disc_fake_loss), linewidth=0.8, color='#1562ff', marker='.')
    plt.title('Discriminator loss fake')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(epoch, smooth_curve_gen(Gen_loss), linewidth=0.8, color='#8a3ac6', marker='.')
    plt.title('Generator loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
