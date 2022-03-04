"""
author: sgdy3
implemention of VRNN
"""

from keras.layers import Conv2D,MaxPool2D,Input,Dense,Reshape,Conv2DTranspose
from keras.models import Sequential,Model
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class VRNN():
    def __init__(self,rnn_size=128):
        self.rows=160
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)

        self.z_dim=20
        self.h_dim=128
        self.x_dim=128

        self.seq_length=10
        self.batch_size=10

        self.rnn_cell = tf.compat.v1.keras.layers.LSTMCell(rnn_size)

    def phi_x(self):
        '''
        feature extraction of input images
        :return: feature vector of img
        '''
        model=Sequential()
        # 160*220*1->80*110*64
        model.add(Conv2D(64,size=5,stride=2,input_shape=self.imgshape),activation='Relu',padding='same')
        # 80*110*64->40*55*32
        model.add(Conv2D(32,size=3,stride=2,input_shape=self.imgshape),activation='Relu',padding='same')
        # 40*55*32->20*22*16
        model.add(Conv2D(16,size=3,stride=2,input_shape=self.imgshape),activation='Relu',padding='same')
        # 20*22*16->10*11*8
        model.add(Conv2D(8,size=3,stride=2,input_shape=self.imgshape),activation='Relu',padding='same')
        # 18*11*8->8
        model.add(MaxPool2D)

        x=Input(shape=self.imgshape)
        feature=model(x)
        return Model(x,feature)

    def phi_z(self):
        '''
        featrue extraction of lantent variable
        :return: feature vector of z
        '''
        model=Sequential()
        model.add(Dense(self.h_dim,input_dim=self.z_dim,activation='Relu'))
        x=Input(shape=self.z_dim)
        feature=model(x)
        return Model(x,feature)

    def phi_prior(self,raw_std_bias=0.25,min_std=1e-5):
        '''
        calculate parameters of p(z_t|h_{t-1})
        :param raw_std_bias: prevent z_std close to 0
        :param min_std: lowest bond of z_std, ensure the variability
        :return: z_mean,log z_var
        '''
        x=Input(shape=self.h_dim)
        outs=Dense(self.h_dim,activation='Relu')(x)
        z_mean=Dense(self.z_dim)(outs)
        z_std=Dense(self.h_dim)(outs) # it seems like all the previous program have't use actiacation func in the last step
        z_std=tf.maximum(tf.nn.softplus(z_std+raw_std_bias),min_std)
        return Model(x,[z_mean,z_std])

    def phi_enc(self,raw_std_bias=0.25,min_std=1e-5):
        '''
        calculate parameters of posterior distribution p(z_t|x_t.h_{t-1})
        :return: z_mean,z_std (posterior)
        '''
        x=Input(shape=self.h_dim+self.x_dim)
        outs=Dense(self.h_dim,activation='Relu')(x)
        outs=Dense(self.h_dim,activation='Relu')(outs)
        enc_mean=Dense(self.z_dim)(outs)
        enc_std=Dense(self.h_dim)(outs)
        enc_std=tf.maximum(tf.nn.softplus(enc_std+raw_std_bias),min_std)
        return Model(x,[enc_mean,enc_std])

    def phi_dec(self):
        '''
        generate new sample x_t from z_t and h_{t-1}
        :return: x_t
        '''
        x=Input(shape=self.z_dim+self.h_dim)
        outs=Dense(10*11*64,activation='relu')(x)
        outs=Reshape(10,11,3)(outs)
        # 10*11*3->20*22*64
        outs=Conv2DTranspose(64,kernel_size=(3,3),strides=(2,2),padding='same',name='convT_1',activation='relu')(outs)
        # 20*22*64->80*110*32
        outs=Conv2DTranspose(32,kernel_size=(3,3),strides=(4,5),padding='same',name='convT_2',activation='relu')(outs)
        # 80*110*32->160*220*16
        outs=Conv2DTranspose(16,kernel_size=(3,3),strides=(2,2),padding='same',name='convT_1',activation='relu')(outs)
        # 160*220*16->160*220*1
        outs=Conv2DTranspose(1,kernel_size=(3,3),strides=(3,3),padding='same',name='convT_1',activation='relu')(outs)
        return Model(x,outs)

    def sampling(self,mean,std):
        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch,dim))
        return mean+std*epsilon

    def train(self,input_images):
        input_feature=self.phi_x()(input_images)
        times=input_images.shape[0]
        batch_size=input_images.shape[1]
        h0 = tf.zeros([times,batch_size,self.rnn_cell.output_size])
        c0 = tf.zeros([times,batch_size,self.rnn_cell.output_size])
        zt_encoded=tf.zeros([times,batch_size,self.z_dim])
        prev_state={'rnn_state':[h0,c0],'zt_ecoded':zt_encoded}
        for t in range(self.seq_length):
            rnn_inputs = tf.concat([input_feature[:, t],prev_state['zt_ecoded']], axis=-1)
            rnn_out,rnn_state=self.rnn_cell(rnn_inputs,prev_state['rnn_state'])
            pri_zmean,pri_zstd=self.phi_prior()(rnn_out)
            post_zmean,post_zstd=self.phi_enc()(tf.concat([input_feature[:, t],rnn_out], axis=-1))
            zt=self.sampling(post_zmean,post_zstd)
            zt_encoded=self.phi_z()(zt)
            generate_x=self.phi_dec()(tf.concat([zt_encoded,rnn_out],axis=-1))
            p_zt=tfp.distributions.MultivariateNormalDiag(loc=pri_zmean,scale_diag=pri_zstd)
            q_zt=tfp.distributions.MultivariateNormalDiag(loc=post_zmean,scale_diag=post_zstd)
