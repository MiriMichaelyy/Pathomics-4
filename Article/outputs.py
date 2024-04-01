import os
import numpy
import inputs
import matplotlib
import matplotlib.pyplot as plt

def save_dataset(path, dataset, offset=0, suffix="png"):
    os.makedirs(path, exist_ok=True)
    count = 0
    for image in dataset:
        count += 1
        image.save(os.path.join(path, f"{offset + count}.{suffix}"))
    return count

def save_losses(path, losses, epoch):
    numpy.save(os.path.join(path, f'E{epoch}_Disc_loss_real.npy'), numpy.array(losses[0]))
    numpy.save(os.path.join(path, f'E{epoch}_Disc_loss_fake.npy'), numpy.array(losses[1]))
    numpy.save(os.path.join(path, f'E{epoch}_Gen_loss.npy'),       numpy.array(losses[2]))

def save_models(path, models, epoch):
    g_model, d_model, gan_model = models
    g_model.save  (os.path.join(path, "models", f"g_model_{epoch}.keras"))
    d_model.save  (os.path.join(path, "models", f"d_model_{epoch}.keras"))
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

def plot_outputs(Disc_real=None, Disc_fake=None, Gen=None):
    if Disc_real is None:
        Disc_real = numpy.load('Disc_loss_real.npy')
    if Disc_fake is None:
        Disc_fake = numpy.load('Disc_loss_fake.npy')
    if Gen is None:
        Gen = numpy.load('Gen_loss.npy')

    Disc_real_loss = []
    Disc_fake_loss = []
    Gen_loss       = []
    for i in range(len(Gen)):
        if i % 450 == 0:
            Gen_loss.append(Gen[i])
            Disc_fake_loss.append(Disc_fake[i])
            Disc_real_loss.append(Disc_real[i])
    
    epoch = range(0, len(Disc_loss_real))
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
