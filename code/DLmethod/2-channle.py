import time
import keras.utils.np_utils
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Dense,GlobalAveragePooling2D,Subtract,Softmax,AvgPool2D
from keras.models import Sequential,Model
from keras.layers import Activation,Dropout,Input,Flatten
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
import keras
import pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from auxiliary.preprocessing import preprocess,hafemann_preprocess
from sklearn.metrics import roc_curve, auc


class TwoC2L():
    def __init__(self):
        self.rows=150
        self.cols=220
        self.channles=2
        self.imgshape = (self.rows, self.cols, self.channles)

        self.batchsize=64
        self.epochs=1


        self.subnet=self.bulid_model()
        self.optimizer= adam_v2.Adam(learning_rate=0.0003)

        sig=Input(shape=self.imgshape)
        label=Input(shape=(1,))

        feature=self.subnet(sig)
        logit1=Dense(2,activation='relu',name='logit1')(feature)
        logit2=Dense(2,activation='relu',name='logit2')(feature)
        subtracted=Subtract(name='sub')([logit1,logit2])
        out=Softmax()(subtracted)

        self.net=Model([sig,label], out)
        self.net.compile(loss='categorical_crossentropy',metrics='accuracy',optimizer=self.optimizer)
        plot_model(self.net, to_file='2-channle.png', show_shapes=True)
        self.net.summary()


    def bulid_model(self):
        model=Sequential()

        model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv1',activation='relu'))
        #model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv2',activation='relu'))
        #model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='pool1'))

        model.add(Conv2D(64, kernel_size=(5, 5), strides=1, padding='same',name='conv3',activation='relu'))
        #model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',name='conv4',activation='relu'))
        #model.add(Activation('relu'))

        model.add(Conv2D(96, kernel_size=(3, 3), strides=1, padding='same',name='conv5',activation='relu'))
        #model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool3'))
        model.add(Dropout(0.3))

        #model.add(Flatten())
        #model.add(Dense(1024,name='fc1'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256,name='fc1'))
        model.add(Dense(128,name='fc2',activation='relu'))
        #model.add(Activation('relu'))

        model.summary()

        img=Input(shape=self.imgshape)
        feature=model(img)
        return  Model(img,feature,name='subnet')


    def train(self,dataset,weights=''):
        save_dir = '../../NetWeights/2C2L_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.net.load_weights(filepath)
        else:
            filepath = os.path.join(save_dir, '2C2L.h5')
            dataset = dataset.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            i=1
            doc=[]
            pre_loss=[]
            min_loss=100 # manually set
            for batch in dataset:
                train_label=keras.utils.np_utils.to_categorical(batch[1])
                loss,acc = self.net.train_on_batch(batch,y=train_label)  # batch[1] are labels
                doc.append(loss)
                print("round %d=> loss:%f, acc:%f%% " % (i,loss,acc*100))
                if(early_stop(20,loss,pre_loss,threshold=0.005)):
                    print("training complete")
                    break
                if(i>500):
                    print("enough rounds!!")
                    break
                i+=1
            self.net.save_weights(filepath)
            return doc

def load_img(file_name1,file_name2,label,ext_h=820,ext_w=890):
    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=hafemann_preprocess(img1.numpy(),ext_h,ext_w)
    img1=np.expand_dims(img1,axis=2)
    # img1=preprocess(img1,ext_h,ext_w)

    img2 = tf.io.read_file(file_name2, 'rb')  # 读取图片
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.rgb_to_grayscale(img2)
    img2=hafemann_preprocess(img2.numpy(),ext_h,ext_w)
    img2=np.expand_dims(img2,axis=2)
    # img2=preprocess(img2,ext_h,ext_w)

    img=tf.concat([img1,img2],axis=-1)
    return img,label

def early_stop(stop_round,loss,pre_loss,threshold=0.005):
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

def curve_eval(label,result):
    fpr, tpr, thresholds = roc_curve(label,result, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size
    area = auc(fpr, tpr)
    print("EER:%f"%EER)
    print('AUC:%f'%area)
    print('ACC(EER_threshold):%f'%acc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()


if __name__=='__main__':
    TC2L=TwoC2L()
    with open('../../pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)
    train_ind=train_ind[np.random.permutation(train_ind.shape[0]),:]
    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1], train_ind[:, 2].astype(np.int8)))

    image = dataset.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.int8]))
    doc=TC2L.train(image)


    mode='test'

    assert  mode=='train' or mode== 'test', 'the programmer can only execute in training or testing model'

    if mode=='train':

        if(doc): # 进行了训练就画图，直接读档模型不画图
            plt.plot(doc)
            plt.title('categorical_crossentropy')
            plt.xlabel('times')
            plt.ylabel('categorical_crossentropy')

        start=time.time()
        result=[]
        label=[]
        cost=[]
        for b in image.batch(40):
            result.append(TC2L.net.predict_on_batch(b))
            label.append(b[1].numpy())
        end=time.time()
        print("time cost : %f"%(end-start))
        result=np.array(result)

        temp=np.zeros((1,2))
        for i in result:
            temp=np.vstack([temp,i]) # 由于batch为32时不能整除，返回result的shape不都是32不能直接化为ndarray
        temp=temp[1:,:]
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        curve_eval(label,result[:,1])

    else:
        with open('../../pair_ind/cedar_ind/test_index.pkl', 'rb') as test_index_file:
            test_ind = pickle.load(test_index_file)
        test_ind = np.array(test_ind)
        test_set= tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1], test_ind[:, 2].astype(np.int8)))
        test_image = test_set.map(
            lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.int8]))

        result=[]
        label=[]

        for b in test_image.batch(40):
            result.append(TC2L.net.predict_on_batch(b))
            label.append(b[1].numpy())
        temp=np.zeros((1,2))
        for i in result:
            temp=np.vstack([temp,i]) # 由于batc可能不能整除，返回result的shape不都是batch大小不能直接化为ndarray
        temp=temp[1:,:]
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        curve_eval(label,result[:,1])

