from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

#reparameterization trick
#z = z_mean + sqrt(var) * eps
def sampling(args):
    """Reparameterization trick by sampling
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    #Arguments:
        args (tensor): mean and log of variance of Q(z|x)
    #Returns:
        z (tensor): sampled latent vector
    """
    z_mean,z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.shape(z_mean)[1]

    epsilon = keras.backend.random_normal(shape=(batch,dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name='vae_mnist'):
    """Plots labels and MNIST digits as function of 2 dim latent vector
    Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    encoder, decoder = models
    x_test, y_test = data
    xmin = ymin = -4
    xmax = ymax = +4
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, 'vae_mean.png')
    # display a 2D plot of the digit classes in the latent space
    z, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))

    # axes x and y ranges
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])

    # subsampling to reduce density of points on the plot
    z = z[0::2]
    y_test = y_test[0::2]
    plt.scatter(z[:, 0], z[:, 1], marker='')
    for i, digit in enumerate(y_test):
        axes.annotate(digit, (z[i, 0], z[i, 1]))
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, 'digits_over_latent.png')
    # display a 30*30 2D mainfold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
    # ?????????????????????????????????????????????????????????????????????
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])  #flatten?????????MLP
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255  #?????????0~1

#?????????
input_shape = original_dim
intermediate_dim = 512  # ?????????????????????512
batch_size = 128
latent_dim = 2  # ?????????????????????
epochs = 50

# VAE model
# encoder (generation model)
# 2???MLP?????????img(x),???????????????(z), p(z|x)????????????????????????z=g(eplison,x),?????????????????????z=u+epslion*sigma,(u,sigma??????encoder???x)
inputs = keras.layers.Input(shape=input_shape,name='encoder_input')
x = keras.layers.Dense(intermediate_dim,activation='relu')(inputs)
z_mean = keras.layers.Dense(latent_dim,name='z_mean')(x)
z_log_var = keras.layers.Dense(latent_dim,name='z_log_var')(x)
z = keras.layers.Lambda(sampling,output_shape=(latent_dim,),name='z')([z_mean,z_log_var])
encoder = keras.Model(inputs,[z_mean,z_log_var,z],name='encoder')

encoder.summary()
keras.utils.plot_model(encoder,to_file='vae_mlp_encoder.png',show_shapes=True)

# decoder (recognitioin model)
# 2???MLP??????????????????z?????????img(x),q(x|z); L=1???????????????Z
latent_inputs = keras.layers.Input(shape=(latent_dim,),name='z_sampling')
x = keras.layers.Dense(intermediate_dim,activation='relu')(latent_inputs)
outputs = keras.layers.Dense(original_dim,activation='sigmoid')(x)
decoder = keras.Model(latent_inputs,outputs,name='decoder')

decoder.summary()
keras.utils.plot_model(decoder,to_file='vae_mlp_decoder.png',show_shapes=True)

# encoder+decoder, ???img??????Z??????Z??????img
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs,outputs,name='vae_mpl')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load tf model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use binary cross entropy instead of mse (default)"
    parser.add_argument("--bce", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.bce:
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    else:
        reconstruction_loss = keras.losses.mse(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    #keras.utils.plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
    save_dir = '../../NetWeights/vae_mlp_weights'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.weights:
        filepath = os.path.join(save_dir, args.weights)
        vae = vae.load_weights(filepath)
    else:
        # train
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        filepath = os.path.join(save_dir, 'vae_mlp.mnist.tf')
        vae.save_weights(filepath)
    plot_results(models, data, batch_size=batch_size, model_name='vae_mlp')
