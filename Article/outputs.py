import os
import numpy
import inputs
import matplotlib
import matplotlib.pyplot as plt

def save_dataset(path, dataset):
    if not os.path.exists(path):
        os.makedirs(path)

    offset = len(os.listdir(path))
    for index, image in enumerate(dataset):
        image.save(f"{path}/{offset + index + 1}.tiff")

def save_losses(losses, path, epoch):
    numpy.save(f'{path}/results/E{epoch}_Disc_loss_real.npy', numpy.array(losses[0]))
    numpy.save(f'{path}/results/E{epoch}_Disc_loss_fake.npy', numpy.array(losses[1]))
    numpy.save(f'{path}/results/E{epoch}_Gen_loss.npy',       numpy.array(losses[2]))

def save_models(models, path, epoch):
    g_model, d_model, gan_model = models

    g_model.save  (f"{path}/models/g_model_{epoch}.keras")
    d_model.save  (f"{path}/models/d_model_{epoch}.keras")
    gan_model.save(f"{path}/models/gan_model_{epoch}.keras")

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

def plot_outputs(style='seaborn', Disc_loss_real=None, Disc_loss_fake=None, Gen_loss=None):
    if Disc_loss_real is None:
        Disc_loss_real = numpy.load('Disc_loss_real.npy')
    if Disc_loss_fake is None:
        Disc_loss_fake = numpy.load('Disc_loss_fake.npy')
    if Gen_loss is None:
        Gen_loss = numpy.load('Gen_loss.npy')

    epoch = range(0, len(Gen_loss))
    matplotlib.style.use(sty)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(epoch, Disc_loss_real, linewidth=0.8, color='#9b0000', marker='.')
    plt.title('Discriminator loss real')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(epoch, smooth_curve_dis(Disc_loss_fake), linewidth=0.8, color='#1562ff', marker='.')
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
