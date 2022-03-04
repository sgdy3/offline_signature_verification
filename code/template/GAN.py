from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Deconv2D
from keras.models import Sequential, Model
from keras.optimizers import adam_v2

import matplotlib.pyplot as plt

import sys
import os
import numpy as np

class DCGAN():
    def __init__(self):
        # 输入shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # 分十类
        self.num_classes = 10
        self.latent_dim = 100
        # adam优化器
        optimizer = adam_v2.Adam(learning_rate=0.0003,beta_1=0.5)
        # 判别模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])
        print("hey, already compiled")
        # 生成模型
        self.generator = self.build_generator()

        # conbine是生成模型和判别模型的结合
        # 判别模型的trainable为False
        # 用于训练生成模型
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        # 先全连接到1024*7*7的维度上
        model.add(Dense(512 * 7 * 7,input_dim=self.latent_dim))
        model.add(Activation("relu"))
        # reshape成特征层的样式
        model.add(Reshape((7, 7,512)))
        # 第一层不加BN

        # 7, 7, 512->14,14,256
        #(14/2)=7
        model.add(Deconv2D(256,kernel_size=(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))

        # 14,14,256 -> 28,28,128
        model.add(Deconv2D(128, kernel_size=(3,3),strides=(2,2), padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))

        # 28, 28, 128 -> 28, 28, 64
        model.add(Deconv2D(64, kernel_size=(3,3),strides=(1,1), padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))


        # 28, 28, 64-> 28, 28, 1
        model.add(Conv2D(self.channels, kernel_size=(3,3),padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        '''
        实际上论文里并没有给出判别器的结构，假定同样是5层吧
        :return:
        '''
        model = Sequential()
        # 28, 28, 1 -> 14, 14, 64
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))

        # 14, 14, 64-> 7, 7, 128
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))

        # 7, 7, 128 -> 4, 4, 256
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))

        # 4，4，512->4,4,1
        model.add(Conv2D(1,kernel_size=3,strides=1,padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))

        # 4，4，1->1
        model.add(GlobalAveragePooling2D())
        # sigmoid
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        # 载入数据
        (X_train, _), (_, _) = mnist.load_data()

        # 归一化
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # --------------------- #
            #  训练判别模型
            # --------------------- #
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # 训练并计算loss
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  训练生成模型
            # ---------------------
            g_loss = self.combined.train_on_batch(noise, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("../../images"):
        os.makedirs("../../images")
    dcgan = DCGAN()
    dcgan.train(epochs=2000, batch_size=256, save_interval=50)

