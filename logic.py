import numpy
import datetime
import outputs

def train(path, models, batch, epoch):
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

        # Generate a batch of fake samples.
        X_fakeB = g_model.predict(X_realA)
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

        # Save the models over the minimum loss generator.
        if g_loss <= min(Gen_loss):
            outputs.save_models(path, models, epoch)

    elapsed_time   = datetime.datetime.now() - start_time
    Disc_loss_real = numpy.array(Disc_loss_real)
    Disc_loss_fake = numpy.array(Disc_loss_fake)
    Gen_loss       = numpy.array(Gen_loss)
    return (d_model, g_model, gan_model), (Disc_loss_real, Disc_loss_fake, Gen_loss)

def normalize(model, image):
    return model.predict(image)