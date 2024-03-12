import keras
from   tensorflow.keras.layers import Input
from   keras.layers            import Conv2D
from   keras.layers            import LeakyReLU
from   keras.layers            import Activation
from   keras.layers            import Concatenate
from   keras.layers            import BatchNormalization

class Discriminator(keras.models.Model):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()

        self.init            = keras.initializers.RandomNormal(stddev=0.02)
        self.in_src_image    = Input(shape=image_shape)
        self.in_target_image = Input(shape=image_shape)
        self.merged          = Concatenate()([self.in_src_image, self.in_target_image])

        self.d   = Conv2D(64,  (4,4), strides=(2,2), padding='same', kernel_initializer=self.init)
        self.d1  = LeakyReLU(alpha=0.2)
        self.d2  = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=self.init)
        self.d3  = BatchNormalization()
        self.d4  = LeakyReLU(alpha=0.2)
        self.d5  = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=self.init)
        self.d6  = BatchNormalization()
        self.d7  = LeakyReLU(alpha=0.2)
        self.d8  = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=self.init)
        self.d9  = BatchNormalization()
        self.d10 = LeakyReLU(alpha=0.2)
        self.d11 = Conv2D(512, (4,4),                padding='same', kernel_initializer=self.init)
        self.d12 = BatchNormalization()
        self.d13 = LeakyReLU(alpha=0.2)
        self.d14 = Conv2D(1,   (4,4),                padding='same', kernel_initializer=self.init)
        self.patch_out = Activation('sigmoid')

    def call(self, inputs):
        x = self.merged(inputs)
        x = self.d(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.d7(x)
        x = self.d8(x)
        x = self.d9(x)
        x = self.d10(x)
        x = self.d11(x)
        x = self.d12(x)
        x = self.d13(x)
        x = self.d14(x)
        return self.patch_out(x)

# Define image shape
image_shape = (256, 256, 3)

# Create an instance of the Discriminator.
discriminator_model = Discriminator(image_shape)

# Compile the model
opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
discriminator_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])