from keras.models                      import Model, Sequential
from keras.layers                      import Input, Dense, Activation
from keras.layers                      import Dropout, Flatten, Reshape
from keras.layers                      import Convolution2D, UpSampling2D
from keras.layers                      import Conv2D, Conv2DTranspose
from keras.layers.normalization        import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers                  import Adam

from keras.datasets                    import mnist
from tqdm                              import tqdm
from PIL                               import Image

import math, os
import numpy             as np
import matplotlib.pyplot as plt

def get_generator( transpose=True ):
        # 1. 100dim noise to 128*7*7dim vec
        # 2. 128*7*7 vec to 128ch-7x7 Mat
        # 3. 128ch-7x7 Mat to 64ch-14x14 Mat by 5x5-2strides filter
        # 4. 64ch-14x14Mat to 1ch-28x28 Mat  by 5x5-2strides filter
    model = Sequential()
    if (transpose):
        model.add( Dense( 128*7*7, input_dim=100) )
        model.add( LeakyReLU(0.2) )
        # 
        model.add( BatchNormalization() )
        model.add( Reshape( (7,7,128) ) )
        model.add( Conv2DTranspose( 64, 5, strides=2, padding="same") )
        model.add( LeakyReLU(0.2) )
        # 
        model.add( BatchNormalization() )
        model.add( Conv2DTranspose( 1, 5, strides=2, padding="same") )
        model.add( Activation('tanh') )
    else:
        model.add( Dense( 128*7*7, input_dim=100) )
        model.add( LeakyReLU(0.2) )
        # 
        model.add( BatchNormalization() )
        model.add( Reshape( (7,7,128) ) )
        model.add( UpSampling2D() )
        model.add( Convolution2D( 64, 5, 5, border_mode="same") )
        model.add( LeakyReLU(0.2) )
        # 
        model.add( BatchNormalization() )
        model.add( UpSampling2D() )
        model.add( Convolution2D( 1, 5, 5, border_mode="same") )
        model.add( Activation('tanh') )
    print(model.summary())
    return model

def get_discriminator():
        # 1. 1ch-28x28 Mat to 64ch-14x14 Mat by 5x5-2strides filter
        # 2. 64ch-14x14 Mat to 128ch-7x7 Mat by 5x5-2strides filter
        # 3. 128ch-7x7 Mat to 128*7*7 vec
        # 4. 128*7*7 vec to scalar
    model = Sequential()
    model.add( Convolution2D(64, 5, 5, subsample=(2,2), input_shape=(28,28,1), border_mode='same') )
    model.add( LeakyReLU(0.2) )
    model.add( Dropout(0.3) )
    # 
    model.add( Convolution2D(128, 5, 5, subsample=(2,2), border_mode='same') )
    model.add( LeakyReLU(0.2) )
    model.add( Dropout(0.3) )
    # 
    model.add( Flatten() )
    model.add( Dense(1) )
    model.add( Activation('sigmoid') )
    return model

class GD_PAIR:
    def __init__(self,generator,discriminator):
        self.g = generator
        self.d = discriminator
    def compileAndBuild(self):
        self.g.compile(loss='binary_crossentropy', optimizer=Adam())
        self.d.compile(loss='binary_crossentropy', optimizer=Adam())
        self.d.trainable = False

        gan_in = Input(shape=(100,))
        x = self.g(gan_in)
        gan_out = self.d(x)

        self.gan = Model(input=gan_in, output=gan_out)
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())

GENERATED_IMAGE_PATH = './generated_images/' # 生成画像の保存先
def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros( (height*rows, width*cols), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image

def save_prediction(i, j, p):
    image = combine_images(p)
    image = image*127.5 + 127.5
    if (not os.path.exists(GENERATED_IMAGE_PATH)):
        os.mkdir(GENERATED_IMAGE_PATH)
    Image.fromarray(image.astype(np.uint8)).save(GENERATED_IMAGE_PATH+"%04d_%04d.png" % (i,j ))


def train(gd, images, epoch=10, batch_size=128):
    batch_count = images.shape[0] // batch_size

    for i in range(epoch):
        for j in tqdm(range(batch_count)):
            noise_input_g = np.random.rand(batch_size, 100)
            image_batch = images[np.random.randint(0, images.shape[0], size=batch_size)]
            predictions = gd.g.predict(noise_input_g, batch_size=batch_size)

            X_discriminator = np.concatenate([predictions, image_batch])
            y_discriminator = [0]*batch_size + [1]*batch_size
            gd.d.trainable = True
            gd.d.train_on_batch(X_discriminator, y_discriminator)

            X_generator = np.random.rand(batch_size, 100)
            y_generator = [1]*batch_size
            gd.d.trainable = False
            gd.gan.train_on_batch(X_generator, y_generator)

            # 生成画像を出力
            if j % 50 == 0:
                save_prediction(i,j,predictions)

            gd.g.save_weights('gen_scaled_images_tmp.h5')
            gd.d.save_weights('dis_scaled_images_tmp.h5')


if(__name__=="__main__"):
    gd_a = GD_PAIR(get_generator(False), get_discriminator())
    gd_a.compileAndBuild()

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5

    gd_a.g.load_weights('gen_scaled_images.h5')
    gd_a.d.load_weights('dis_scaled_images.h5')

    train(gd_a, X_train, 30, 128)
    gd_a.g.save_weights('gen_scaled_images_30.h5')
    gd_a.d.save_weights('dis_scaled_images_30.h5')
    train(gd_a, X_train, 20, 128)
    gd_a.g.save_weights('gen_scaled_images_50.h5')
    gd_a.d.save_weights('dis_scaled_images_50.h5')

