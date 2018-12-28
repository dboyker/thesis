import sys
sys.path.append("..")
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
import numpy as np
from Helper import data_loader


MODELS_PATH = '../Models/GAN/'
RESULTS_PATH = '../Results/GAN/'


class GAN():

    def __init__(self, data):
        self.data = data
        self.img_shape = (data.img_size, data.img_size, 1)
        self.latent_dim = 100
        self.optimizer = Adam(0.0002, 0.5)
        self.build_discriminator()  # Build and compile the discriminator
        self.discriminator.trainable = False
        self.build_generator()
        self.discriminator.trainable = True  # To prevent warning

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        self.generator = Model(noise, img)
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.combined = Model(z, validity)  # Train the generator to fool the discriminator
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        self.discriminator = Model(img, validity)
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])

    def train_generator(self, batch_size, valid):
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        g_loss = self.combined.train_on_batch(noise, valid)
        return g_loss

    def train_discriminator(self, batch_size, valid, fake, x_train):
        # Train the discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)   # Select a random batch of images
        imgs = x_train[idx]
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))  # Random noise
        gen_imgs = self.generator.predict(noise)  # Generate a batch of new images
        d_loss_real = self.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss

    def train(self, epochs, batch_size=128, sample_interval=50):
        x_train = self.data.x_train
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            d_loss = self.train_discriminator(batch_size, valid, fake, x_train)
            g_loss = self.train_generator(batch_size, valid)
            self.save_results(epoch, d_loss[0], 100*d_loss[1], g_loss, sample_interval)

    def save_results(self, epoch, d_loss, acc, g_loss, sample_interval):
        #print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        if epoch % 100 == 0:
            with open(RESULTS_PATH + 'loss_log.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([epoch, d_loss, acc, g_loss])
        if epoch % sample_interval == 0:  # Save interval
            self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images 0 - 1
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Vanilla Gan Images/%d.png" % epoch)
        plt.close()


def load_gan(model):
    model.discriminator.load_weights(MODELS_PATH + 'discriminator_w.h5')
    model.combined.load_weights(MODELS_PATH + 'combined_w.h5')


def save_gan(model):
    model.discriminator.save_weights(MODELS_PATH + 'discriminator_w.h5')
    model.combined.save_weights(MODELS_PATH + 'combined_w.h5')


if __name__ == '__main__':
    data = data_loader.DataStructure('gan')
    model = GAN(data)
    try:
        load_gan(model)
        model.sample_images(epoch=10)
    except OSError:
        model.train(epochs=40, batch_size=32, sample_interval=200)
        save_gan(model)