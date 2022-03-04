from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.optimizers import adam_v2
'''
图片太大了，有没有可能分成小块来处理呢
---------------2021.10.31--------------
记得补上preprocess模块，影响很大
'''


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
                 n_dim,
                 model_name='vae_sig'):
    """Plots labels and MNIST digits as function of 2 dim latent vector
    Arguments:
        models (tuple): encoder and decoder models
        data (dataset): test data and label
        n_dim(int): dim of lantent variables
        model_name (string): which model is using this function
    """
    encoder, decoder = models
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, 'vae_mean_feature.png')
    # display a 2D plot of the digit classes in the latent space
    z=[]
    labels=[]
    for img,label in data.batch(40):
        temp_z,_,_=encoder.predict_on_batch(img)
        z.append(temp_z)
        labels.append(label)
    z=np.vstack(z)
    labels=np.array(labels).reshape(-1,)
    plt.figure(figsize=(12, 10))

    axes = plt.gca()
    pca_z=PCA(z)
    pca_z.fit()
    pca_z.reduce(2)  # decompcose z to 2-dim for visualizaion
    plt.scatter(z[:, 0], z[:, 1], marker='')
    for i, signer in enumerate(labels):
        axes.annotate(signer, (z[i, 0], z[i, 1]))
    plt.xlabel('pca_z[0]')
    plt.ylabel('pca_z[1]')
    plt.savefig(filename)
    plt.show()


def train_preprocess(file_name):
    img = tf.io.read_file(file_name, 'rb')  # 读取图片
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [160, 220])  # the dataype will change to float32 because of inteplotation
    img = tf.cast(img,tf.uint8)
    img=255-img #invert
    img=img.numpy() # haven't found method to appply boolean index or threshold with tf
    img[img>=50]=255
    img[img < 50]=0
    img=tf.convert_to_tensor(img)
    img=tf.cast(img,tf.float32)
    img=img/255
    return img

def test_preprocess(file_name,label):
    img = tf.io.read_file(file_name, 'rb')  # 读取图片
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [160, 220])  # the dataype will change to float32 because of inteplotation
    img = tf.cast(img,tf.uint8)
    img=255-img #invert
    img=img.numpy() # haven't found method to appply boolean index or threshold with tf
    img[img>=50]=255
    img[img < 50]=0
    img=tf.convert_to_tensor(img)
    img=tf.cast(img,tf.float32)
    img=img/255
    return img,label

def load_data(path):
    org_author = 55
    org_num = 24
    sig_ind=[]
    for i in range(1,org_author+1):
        for j in range(1,org_num+1):
            sig_ind.append(path%(i,j))
    data=tf.data.Dataset.from_tensor_slices(sig_ind)
    data=data.map(lambda x:tf.py_function(func=train_preprocess, inp=[x], Tout=[tf.float32]))
    return data

class sig_vae():
    def __init__(self):
        self.height=160
        self.width=220
        self.channels=1
        self.img_shape=(self.height,self.width,self.channels)

        self.batch_size=64
        self.epochs=50

        # VAE model
        # encoder (generation model)
        # 2层MLP，输入img(x),输出隐变量(z), p(z|x)用重参数来拟合，z=g(eplison,x),这里是用正态，z=u+epslion*sigma,(u,sigma对应encoder的x)
        # encoder+decoder, 由img估计Z，由Z重构img
        self.lantent_dim=4 # dim of z
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()

        sig_img=keras.layers.Input(shape=self.img_shape)
        z_mean,z_logvar,z=self.encoder(sig_img)
        sig_sys = self.decoder(z)

        kl_loss = 1 + z_logvar - keras.backend.square(z_mean) - keras.backend.exp(z_logvar)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        self.vae=keras.models.Model(sig_img,sig_sys,name='vae_cnn')
        self.model_loss(kl_loss)


    def build_encoder(self):
        encoder_inputs = keras.layers.Input(shape=self.img_shape, name='encoder_input')
        x = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, name='conv1')(encoder_inputs)
        x = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, name='conv2')(x)
        x = keras.layers.Activation('relu')(x)
        x=keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(8, kernel_size=(3, 3), strides=2, name='conv3')(x)
        x = keras.layers.Activation('relu')(x)
        x=keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(100, activation='relu')(x)
        z_mean = keras.layers.Dense(self.lantent_dim, name='z_mean')(x)
        z_log_var = keras.layers.Dense(self.lantent_dim, name='z_log_var')(x)
        z = keras.layers.Lambda(sampling, output_shape=(self.lantent_dim,), name='z')([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        keras.utils.plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        return encoder

    def build_decoder(self):
        # decoder (recognitioin model)
        # 2层MLP，输入隐变量z，输出img(x),q(x|z); L=1，只取一组Z
        latent_inputs = keras.layers.Input(shape=(self.lantent_dim,), name='z_sampling')
        x = keras.layers.Dense(10*11*64, activation='relu')(latent_inputs)
        x = keras.layers.Reshape((10,11,64))(x)
        x = keras.layers.Conv2DTranspose(128,kernel_size=(3,3),strides=(2,2),padding='same',name='convT_1',activation='relu')(x)
        x = keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(4,5), padding='same',name='convT_2', activation='relu')(x)
        x = keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2,2), padding='same',name='convT_3', activation='relu')(x)
        decoder_outputs = keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid')(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        decoder.summary()
        keras.utils.plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
        return decoder

    def model_loss(self,kl_loss,bce=True):
        # VAE loss = mse_loss or xent_loss + kl_loss
        input_f=keras.layers.Flatten()(self.vae.input)
        output_f=keras.layers.Flatten()(self.vae.output)
        if bce:
            reconstruction_loss = keras.losses.binary_crossentropy(input_f, output_f)
        else:
            reconstruction_loss = keras.losses.mse(input_f, output_f)
        reconstruction_loss *= self.height*self.width
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=adam_v2.Adam(learning_rate=1e-3))

    def train_model(self,data,weights=''):
        save_dir = '../../NetWeights/Sigvae_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.vae.load_weights(filepath)
        else:
            # train
            times = 0
            pre_loss=[]
            min_loss=4500 # manually set
            filepath = os.path.join(save_dir, 'sigvae.h5')
            for batch in data.shuffle(200).repeat(self.epochs).batch(self.batch_size):
                loss = self.vae.train_on_batch(batch)
                print("%d loss: %f " % (times, loss))
                times += 1
                if(loss<min_loss and times>300): # to avoid frequently saving in the early stage
                    print('save')
                    min_loss=loss
                    self.vae.save_weights(filepath)
                if(times%20==0):  # learning rate decay
                    lr=self.vae.optimizer.lr
                    print(lr)
                    self.vae.optimizer.lr=lr-0.01*lr
                if(early_stop(50,loss,pre_loss,threshold=0.2)):
                    break


def early_stop(stop_round,loss,pre_loss,threshold=0.02):
    '''
    early stop setting
    :param stop_round: rounds under caculated
    :param pre_loss: loss list
    :param threshold: minimum one-order value of loss list
    :return: whether or not to jump out
    '''
    if(len(pre_loss)<stop_round):
        pre_loss.append(loss)
        return False
    else:
        loss_diff=np.diff(pre_loss,1)
        pre_loss.pop(0)
        pre_loss.append(loss)
        if(abs(loss_diff).mean()<threshold): # to low variance means flatten field
            return True
        else:
            return False



class PCA():
    def __init__(self,data):
        self.x=data
        self.sigma=None
        self.singular_values=None
        self.u=None


    def fit(self):
        singular_values,u,_=tf.linalg.svd(self.x)
        sigma=tf.linalg.diag(singular_values)

        self.singular_values=singular_values
        self.sigma=sigma
        self.u=u

    def reduce(self,n_dimensions):
        sigma = tf.slice(self.sigma, [0, 0], [self.x.shape[1], n_dimensions])
        pca = tf.matmul(self.u, sigma)

        normalized_singular_values = self.singular_values / sum(self.singular_values)
        ladder = np.cumsum(normalized_singular_values)
        keep_info=ladder[n_dimensions-1]
        print("keep %f%% imformation "%(keep_info*100))
        return  pca


if __name__ == '__main__':
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    sys_org_path=r'E:\material\signature\signatures\sys_org\sys_org_%d_%d.png'
    sys_forg_path=r'E:\material\signature\signatures\sys_forg\sys_forg_%d_%d.png'

    sig_mod=sig_vae()
    data = load_data(org_path)
    sig_mod.train_model(data)
    result=[]
    for b in data.batch(40):
        z=sig_mod.encoder.predict_on_batch(b)
        result.append(sig_mod.decoder.predict_on_batch(z[0]))
    result=np.vstack(result)
    for i,img in enumerate(result):
        author=i//24+1
        num=i-(author-1)*24
        save_path=(sys_org_path %(author,num))
        img=tf.cast(img*255,tf.uint8)
        img=tf.image.encode_png(img)
        with tf.io.gfile.GFile(save_path,'wb') as file:
            file.write((img.numpy()))

    forg_data = load_data(org_path)
    result=[]
    for b in forg_data.batch(40):
        z=sig_mod.encoder.predict_on_batch(b)
        result.append(sig_mod.decoder.predict_on_batch(z[0]))
    result=np.vstack(result)
    for i,img in enumerate(result):
        author=i//24+1
        num=i-(author-1)*24
        save_path=(sys_forg_path %(author,num))
        img=tf.cast(img*255,tf.uint8)
        img=tf.image.encode_png(img)
        with tf.io.gfile.GFile(save_path,'wb') as file:
            file.write((img.numpy()))


    '''
    org_author = 10
    org_num = 24
    sig_ind=[]
    label=[]
    for i in range(1,org_author+1):
        for j in range(1,org_num+1):
            sig_ind.append(org_path%(i,j))
            label.append(i)
    test_image=tf.data.Dataset.from_tensor_slices((sig_ind,label))
    image=test_image.map(lambda x,y:tf.py_function(func=test_preprocess, inp=[x,y], Tout=[tf.float32,tf.int32]))

    
    plot_results((sig_mod.encoder,sig_mod.decoder), image, 4,model_name='vae_cnn')

    temp=data.shuffle(200).take(1)
    for i in temp.batch(1):
        example=i
    z=sig_mod.encoder.predict(example)
    forg_img=sig_mod.decoder.predict(z[2])
    forg_img=np.squeeze(forg_img)
    example=np.squeeze(example)
    plt.figure(2)
    plt.imshow(forg_img)
    plt.figure(3)
    plt.imshow(example)
    '''
