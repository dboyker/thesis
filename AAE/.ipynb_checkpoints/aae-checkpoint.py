from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge,  Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras import losses
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
sns.set(font_scale=1.5)
p = sns.color_palette("Set2")
sns.palplot(p)
sns.set_palette(p)
cmap = plt.cm.get_cmap('Spectral')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 3})

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)
x_test = (x_test.astype(np.float32) - 127.5) / 127.5
x_test = np.expand_dims(x_test, axis=3)


class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 8

        #optimizer = Adam(0.0002, 0.5)
        #lr = 0.0001
        #mm = 0.5
        #optimizer_discriminator = SGD(lr=lr, momentum=mm, nesterov=False)
        optimizer_discriminator = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_discriminator,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)
        lr = 0.0001
        mm = 0.5
        #optimizer_reconstruction = SGD(lr=lr, momentum=mm, nesterov=False)
        optimizer_reconstruction = Adam(0.0002, 0.5)
        
        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer_reconstruction)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
    def build_encoder(self):
        # Encoder

        img = Input(shape=self.img_shape)
        h = Flatten()(img)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([mu, log_var])
        return Model(img, latent_repr)

    def build_decoder(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)
      
    def update_lr(self, epoch):
        #K.get_value(self.discriminator.optimizer.lr)
        if epoch == 50:
            print('Epoch 50: new lr')
            K.set_value(self.discriminator.optimizer.lr, 0.01)
            K.set_value(self.adversarial_autoencoder.optimizer.lr, 0.001)
        if epoch == 1000:
            print('Epoch 1000: new lr')
            K.set_value(self.discriminator.optimizer.lr, 0.001)
            K.set_value(self.adversarial_autoencoder.optimizer.lr, 0.0001)

    def train(self, epochs, batch_size=128, sample_interval=50):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(1, epochs+1):
            #self.update_lr(epoch)
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            acc_real = d_loss_real[1] * 100
            acc_fake = d_loss_fake[1] * 100
            acc = 0.5 * np.add(acc_real, acc_fake)
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])[0]
            if (epoch % 5 == 0) or (epoch == 1):
                with open('log.csv', 'a') as f:
                    f.write('%d;%f;%f;%f;%f;%f\n' % (epoch, d_loss, acc, acc_real, acc_fake, g_loss))
            if (epoch % sample_interval == 0) or (epoch == 1) or (epoch == epochs+1):
                self.sample_images(epoch)
                print ("e: %d d_l: %f acc: %f acc_r: %f acc_f: %f g_l: %f" % (epoch, d_loss, acc, acc_real, acc_fake, g_loss))
            if (epoch == epochs):
                self.save(epoch)
                
    def sample_images(self, epoch):
        r, c = 5, 5
        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("mnist_%d.png" % epoch)
        plt.close()

    def save(self, epoch):
        self.discriminator.save('aae_discriminator_m_%d.h5' % (epoch))
        self.adversarial_autoencoder.save('aae_m_%d.h5' % (epoch))
        self.encoder.save('aae_encoder_m_%d.h5' % (epoch))
        self.decoder.save('aae_decoder_m_%d.h5' % (epoch))