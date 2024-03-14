import tensorflow
import keras
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal

class Generator(keras.models.Model):
    def __init__(self, image_shape=(256,256,3)):
        super(Generator, self).__init__()

        # Weight initialization
        init = RandomNormal(stddev=0.02)

        # Define image input
        self.in_image = tensorflow.keras.layers.Input(shape=image_shape)

        # Encoder model
        e1 = self.define_encoder_block(self.in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)

        # Bottleneck
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)

        # Decoder model
        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)

        # Output
        g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        self.out_image = Activation('tanh')(g)

        # Define model
        self.model = keras.models.Model(self.in_image, self.out_image)

    def call(self, inputs):
        return self.model(inputs)

    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):
        # Add downsampling layer
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(layer_in)

        # Conditionally add batch normalization
        if batchnorm:
            g = BatchNormalization()(g, training=True)

        # Leaky ReLU activation
        g = LeakyReLU(alpha=0.2)(g)
        return g

    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        # Add upsampling layer
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(layer_in)

        # Add batch normalization
        g = BatchNormalization()(g, training=True)

        # Conditionally add dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)

        # Merge with skip connection
        g = Concatenate()([g, skip_in])

        # ReLU activation
        g = Activation('relu')(g)
        return g