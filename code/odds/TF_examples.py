import keras.losses
import numpy as np
import tensorflow as tf
from keras import  Model
from keras.layers import Conv2D,BatchNormalization,Flatten,Dense,Input,GlobalAveragePooling2D,MaxPooling2D
import keras.backend as K
import keras
import keras.optimizers
import keras.metrics
import keras.datasets.mnist

'''
============================================================
模型构造及训练方法验证：
1. Sequential()方法
2. Model()方法
3. Subclass()方法
模型训练方法：
1. fit()=train_on_batch()
2. GradientTape()
============================================================
'''





# Construction method 1: Subclass()

class MyModel(Model):

    def __init__(self):

        super(MyModel, self).__init__()

        self.conv1 = Conv2D(64, 3, activation='relu')

        self.batch_norm1=BatchNormalization()

        self.pool1=MaxPooling2D()

        self.conv2=Conv2D(64,3,activation='relu')

        self.batch_norm2=BatchNormalization()

        self.pool2=MaxPooling2D()

        self.flatten = Flatten()

        self.d1 = Dense(128, activation='relu')

        self.d2 = Dense(10, activation='softmax')



    def call(self, x,training=True):

        x = self.conv1(x)

        x = self.batch_norm1(x,training=training)

        x=self.pool1(x)

        x=self.conv2(x)

        x=self.batch_norm2(x,training=training)

        x=self.pool2(x)

        x = self.flatten(x)

        x = self.d1(x)

        return self.d2(x)


def get_bn_vars(collection):

    moving_mean, moving_variance = None, None
    for var in collection:

        name = var.name.lower()

        if "variance" in name:

            moving_variance = var

        if "mean" in name:

            moving_mean = var



    if moving_mean is not None and moving_variance is not None:

        return moving_mean, moving_variance

    raise ValueError("Unable to find moving mean and variance")


class TrainStage():
    def __init__(self):
        self.optimizer=keras.optimizers.rmsprop_v2.RMSProp()
        self.loss=keras.losses.SparseCategoricalCrossentropy()
        self.acc=keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self,model,image,label):

        with tf.GradientTape() as tape:

            predictions = model(image)

            loss = self.loss(label,predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        acc=self.acc(label,predictions)
        return loss,acc


model = MyModel()
model.build((None,28,28,1))
model.call(Input(shape=(28,28,1)))
model.summary()

# verifying data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train=np.expand_dims(x_train,axis=-1)
x_train=x_train.astype(np.float32)
train_batches=tf.data.Dataset.from_tensor_slices((x_train,y_train))

# training method 1
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),metrics=keras.metrics.SparseCategoricalAccuracy(),optimizer=keras.optimizers.rmsprop_v2.RMSProp())
model.fit(x_train,y_train,batch_size=32,epochs=1)


# training method 2
train_stage=TrainStage()
iter=0
for images,labels in train_batches.batch(32):
    loss,acc=train_stage.train_step(model,images,labels)
    print(f"iter:{iter}:loss---->{loss},acc----->{acc}")
    # mean, variance = get_bn_vars(model.variables)
    # print(f'iter{iter}:mean-->{mean},var--->{variance}')
    iter+=1

# training method 3
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),metrics=keras.metrics.SparseCategoricalAccuracy(),optimizer=keras.optimizers.rmsprop_v2.RMSProp())
iter=0
for images,labels in train_batches.batch(32):
    loss,acc=model.train_on_batch(images,labels)
    print(f"iter:{iter}:loss========>{loss}-------ACC==========>{acc}")
    iter+=1


