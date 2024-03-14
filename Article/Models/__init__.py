from Models.discriminator import define_discriminator
from Models.generator     import define_generator
from Models.gan           import define_gan

def define_models(shape):
    d_model   = define_discriminator(shape)
    g_model   = define_generator(shape)
    gan_model = define_gan(g_model, d_model, shape)

    return d_model, g_model, gan_model