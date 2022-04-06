# -*- coding: utf-8 -*-
# ---
# @File: Inception.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/28
# Describe: Inception网络的实施
# ---
'''
2022/04/03
-------------
开始动手看代码了，
摘自官方实现的谱系：
  Here is a mapping from the old_names to the new names:
  Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

代码参考来源：
官方：https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
代码架构来源：https://github.com/calmisential/TensorFlow2.0_InceptionV3/blob/master/models/inception_modules.py
网络结构参考：https://github.com/mindspore-ai/models/blob/master/official/cv/inceptionv3/src/inception_v3.py
              https://blog.csdn.net/Student_PAN/article/details/105246485
'''



# from tensorflow import keras
# from tensorflow.keras import Model
# from tensorflow.python import keras
# import tensorflow as tf
# tf.keras

import keras
import keras.layers



class ConvBN(keras.layers.Layer):
    '''
    Inception_v3中最基本的模块，包括CONV,BN,RELU
    '''
    def __init__(self, filters, kernel_size, strides, padding):
        super(ConvBN, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        output = self.conv(inputs)
        output = self.bn(output, training=training)
        output = self.relu(output)

        return output

class Inception_A(keras.layers.Layer):
    """
    3xInception的模板
    """
    def __init__(self,pool_out):
        super(Inception_A, self).__init__()

        self.b0=ConvBN(filters=64,kernel_size=(1,1),strides=1,padding="same")
        self.b1=keras.Sequential([
            ConvBN(filters=48,kernel_size=(1,1),strides=1,padding="same"),
            ConvBN(filters=64,kernel_size=(5,5),strides=1,padding="same")
            ]
        )
        self.b2=keras.Sequential([
            ConvBN(filters=64,kernel_size=(1,1),strides=1,padding="same"),
            ConvBN(filters=96,kernel_size=(3,3),strides=1,padding="same"),
            ConvBN(filters=96,kernel_size=(1,1),strides=1,padding="same")
            ]
        )
        self.b3=keras.Sequential([
            keras.layers.AvgPool2D(pool_size=(3,3),strides=1,padding='same'),
            ConvBN(filters=pool_out,kernel_size=(1,1),strides=1,padding="same")
            ]
        )

    def call(self,inputs,training=None,**kwargs):
        b0_out=self.b0(inputs,training=training)
        b1_out=self.b1(inputs,training=training)
        b2_out=self.b2(inputs,training=training)
        b3_out=self.b3(inputs,training=training)

        out=keras.layers.concatenate([b0_out,b1_out,b2_out,b3_out],axis=-1)

        return out


class Inception_B(keras.layers.Layer):
    '''
    Reduction模块
    '''
    def __init__(self):
        super(Inception_B, self).__init__()

        self.b0=ConvBN(filters=384,kernel_size=(3,3),strides=2,padding="valid")
        self.b1=keras.Sequential([
            ConvBN(filters=64,kernel_size=(1,1),strides=1,padding="same"),
            ConvBN(filters=96,kernel_size=(3,3),strides=1,padding="same"),
            ConvBN(filters=96,kernel_size=(3,3),strides=1,padding="same")
        ]
        )
        self.b2=keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding="valid")

    def call(self,inputs,training=None,**kwargs):
        b0_out=self.b0(inputs,training=training)
        b1_out=self.b1(inputs,training=training)
        b2_out=self.b2(inputs,training=training)

        out=keras.layers.concatenate([b0_out,b1_out,b2_out],axis=-1)

        return out

class Inception_C(keras.layers.Layer):
    '''
    5xInception中的模板
    '''
    def __init__(self,channel_7):
        super(Inception_C, self).__init__()

        self.b0=ConvBN(filters=192,kernel_size=(1,1),strides=1,padding="same")
        self.b1=keras.Sequential([
            ConvBN(filters=channel_7,kernel_size=(1,1),strides=1,padding="same"),
            ConvBN(filters=channel_7,kernel_size=(1,7),strides=1,padding="same"),
            ConvBN(filters=192,kernel_size=(7,1),strides=1,padding="same")
        ]
        )
        self.b2=keras.Sequential([
            ConvBN(filters=channel_7,kernel_size=(1,1),strides=1,padding="same"),
            ConvBN(filters=channel_7,kernel_size=(7,1),strides=1,padding="same"),
            ConvBN(filters=channel_7,kernel_size=(1,7),strides=1,padding="same"),
            ConvBN(filters=channel_7,kernel_size=(7,1),strides=1,padding="same")
        ]
        )
        self.b3=keras.Sequential(
            keras.layers.AvgPool2D(pool_size=(3,3),strides=1,padding="same"),
            ConvBN(filters=192,kernel_size=(1,1),strides=1,padding="same")
        )

    def call(self,inputs,training=None,**kwargs):
        b0_out=self.b0(inputs,training=training)
        b1_out=self.b1(inputs,training=training)
        b2_out=self.b2(inputs,training=training)
        b3_out=self.b3(inputs,training=training)

        out=keras.layers.concatenate([b0_out,b1_out,b2_out,b3_out],axis=-1)

        return out

class Inception_D(keras.layers.Layer):
    '''
    Reduction 模块
    '''
    def __init__(self):
        super(Inception_D, self).__init__()

        self.b0=keras.Sequential([
            ConvBN(filters=192,kernel_size=(1,1),strides=1,padding="same"),
            ConvBN(filters=320,kernel_size=(3,3),strides=2,padding="valid")
        ]
        )
        self.b1=keras.Sequential([
            ConvBN(filters=192,kernel_size=(1, 1),strides=1,padding="same"),
            ConvBN(filters=192,kernel_size=(1, 7),strides=1,padding="same"),
            ConvBN(filters=192,kernel_size=(7, 1),strides=1,padding="same"),
            ConvBN(filters=192,kernel_size=(3, 3),strides=2,padding="same"),

        ])
        self.b3=keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='valid')

    def call(self,inputs,training=None,**kwargs):
        b0_out=self.b0(inputs,training=training)
        b1_out=self.b1(inputs,training=training)
        b2_out=self.b2(inputs,training=training)

        out=keras.layers.concatenate([b0_out,b1_out,b2_out],axis=-1)

        return out

class Inception_E(keras.layers.Layer):
    '''
    2xInception的模板
    '''
    def __init__(self):
        super(Inception_E, self).__init__()

        self.b0= ConvBN(filters=320,kernel_size=(1, 1),strides=1,padding="same")
        self.b1= ConvBN(filters=384,kernel_size=(1, 1),strides=1,padding="same")
        self.b1_a=ConvBN(filters=384,kernel_size=(1,3),strides=1,padding="same")
        self.b1_b=ConvBN(filters=384,kernel_size=(3,1),strides=1,padding="same")

        self.b2=keras.Sequential([
            ConvBN(filters=448,kernel_size=(1, 1),strides=1,padding="same"),
            ConvBN(filters=384,kernel_size=(3, 3),strides=1,padding="same"),
        ])

        self.b2_a=ConvBN(384,kernel_size=(1,3),strides=1,padding="same")
        self.b2_b=ConvBN(384,kernel_size=(3,1),strides=1,padding="same")

        self.b4=keras.Sequential([
            keras.layers.AvgPool2D(pool_size=(3,3),strides=1,padding="same"),
            ConvBN(192,kernel_size=(1,1),strides=1,padding="same")
        ]
        )

    def call(self,inputs,training=None,**kwargs):
        b0_out=self.b0(inputs,training=training)

        b1=self.b1(inputs,training=training)
        b1_out=keras.layers.concatenate([self.b1_a(b1,training=training),self.b1_b(b1,training=training)],axis=-1)

        b2=self.b2(inputs,training=training)
        b2_out=keras.layers.concatenate([self.b2_a(b2,training=training),self.b2_b(b2,training=training)],axis=-1)

        b3_out=self.b3(inputs,training=training)
        out=keras.layers.concatenate([b0_out,b1_out,b2_out,b3_out],axis=-1)

        return out


class Logits(keras.layers.Layer):
    '''
    没有softamx激活啊，损失函数要选对。
    '''
    def __init__(self, num_classes=10, dropout_keep_prob=0.8):
        super(Logits, self).__init__()
        self.avg_pool = keras.layers.AvgPool2D(pool_size=(8,8),padding="valid")
        self.dropout = keras.layers.Dropout(keep_prob=dropout_keep_prob)
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(2048, num_classes)

    def call(self, x,training=None,**kwargs):
        x = self.avg_pool(x)
        x = self.dropout(x,training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InceptionV3(keras.Model):
    def __init__(self,num_class=10,**kwargs):
        super(InceptionV3, self).__init__()
        self.Conv2d_1a_3x3=ConvBN(32,kernel_size=(3,3),strides=2,padding='valid')
        self.Conv2d_2a_3x3=ConvBN(32,kernel_size=(3,3),strides=1,padding="valid")
        self.Conv2d_2b_3x3=ConvBN(64,kernel_size=(3,3),strides=1,padding="same")
        self.Max_Pool1=keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.Conv2d_3b_1x1=ConvBN(80,kernel_size=(1,1),strides=1,padding="same")
        self.Conv2d_4a_3x3=ConvBN(192,kernel_size=(3,3),strides=1,padding="valid")
        self.Max_Pool2=keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.Mixed_5b=Inception_A(pool_out=32)
        self.Mixed_5c=Inception_A(pool_out=64)
        self.Mixed_5d=Inception_A(pool_out=64)
        self.Mixed_6a=Inception_B()
        self.Mixed_6b=Inception_C(channel_7=128)
        self.Mixed_6c=Inception_C(channel_7=160)
        self.Mixed_6d=Inception_C(channel_7=160)
        self.Mixed_6e=Inception_C(channel_7=192)
        self.Mixed_7a=Inception_D()
        self.Mixed_7b=Inception_E()
        self.Mixed_7c=Inception_E()
        self.logits=Logits(num_class)

    def call(self,input,training=None, mask=None):
        out=self.Conv2d_1a_3x3(input,training=None)
        out=self.Conv2d_2a_3x3(out,training=None)
        out=self.Conv2d_2b_3x3(out,training=None)
        out=self.Max_Pool1(out,training=None)
        out=self.Conv2d_3b_1x1(out,training=None)
        out=self.Conv2d_4a_3x3(out,training=None)
        out=self.Max_Pool2(out,training=None)
        out = self.Mixed_5b(out,training=None)
        out = self.Miouted_5c(out,training=None)
        out = self.Miouted_5d(out,training=None)
        out = self.Miouted_6a(out,training=None)
        out = self.Miouted_6b(out,training=None)
        out = self.Miouted_6c(out,training=None)
        out = self.Miouted_6d(out,training=None)
        out = self.Miouted_6e(out,training=None)
        out = self.Miouted_7a(out,training=None)
        out = self.Miouted_7b(out,training=None)
        out = self.Miouted_7c(out,training=None)
        out=self.logit(out,training=None)

        return out



