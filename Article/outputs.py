import numpy
import inputs
import matplotlib
import matplotlib.pyplot as plt

def summarize_performance(path, epoch, batch, g_model, n_samples=3):
    # select a sample of input images
    [X_realB, X_realA] = inputs.load_batch(f"{path}/train", n_samples)
    # generate a batch of fake samples
    X_fakeB = g_model.predict(X_realA)

    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    # plot real source images
    if not os.path.exists(f"{path}/results")
        os.makedirs(f"{path}/results")

    for i in range(n_samples):
        plt.imsave(f"{path}/results/Generated_E{epoch+1}_B{batch+1}_{i}.tiff", X_fakeB[i])
        # plt.imsave(os.path.join(path, "results", 'Generated_E%d_B%d_%d.tiff' % (epoch + 1, batch + 1, i + 1), X_fakeB[i]))

    # save the generator model
    if not os.path.exists(f"{path}/models"):
        os.makedirs(f"{path}/models")
    g_model.save(f"{path}/models/model_{epoch+1}_{batch+1}.h5")

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

def plot_outputs(style='seaborn'):
    Gen_loss       = numpy.load('Gen_loss.npy')
    Disc_loss_real = numpy.load('Disc_loss_real.npy')
    Disc_loss_fake = numpy.load('Disc_loss_fake.npy')

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