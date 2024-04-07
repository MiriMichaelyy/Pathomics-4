import os
import numpy
import imageio
import datetime

def normalize(model, image):
    return model.predict(image)

def train(models, batch):
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
    for i, (X_realB, X_realA) in enumerate(batch):

        # Generate a batch of fake samples.i
        X_fakeB = g_model.predict(X_realA)
        exit()

        y_fake  = numpy.zeros((len(X_fakeB), n_patch, n_patch, 1))

        # Update models for real samples.
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        # Save the loss values in the array
        Disc_loss_real.append(d_loss1)
        Disc_loss_fake.append(d_loss2)
        Gen_loss.append(g_loss)

    elapsed_time = datetime.datetime.now() - start_time
    Disc_loss_real = numpy.array(Disc_loss_real)
    Disc_loss_fake = numpy.array(Disc_loss_fake)
    Gen_loss       = numpy.array(Gen_loss)
    return (g_model, d_model, gan_model), (Disc_loss_real, Disc_loss_fake, Gen_loss)

def test(results_path, model, dataset):
    results_path = os.path.join(results_path, "test", "generated", "%d.tiff")
    start_time = datetime.datetime.now()
    for index, (color, grayscale) in enumerate(dataset):
        print(f"Testing image #{index+1}")
        generated = numpy.squeeze(numpy.array(model.predict(grayscale)))
        imageio.imwrite(results_path % (index + 1), generated)
    print('time: ', datetime.datetime.now() - start_time)