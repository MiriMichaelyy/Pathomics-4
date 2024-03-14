import numpy
import datetime

import inputs
import outputs

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