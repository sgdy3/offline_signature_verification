import time

import cv2 as cv
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Dense,GlobalAveragePooling2D,Lambda
from keras.models import Sequential,Model
from keras.layers import Activation,Dropout,Input,Flatten
from keras.optimizers import adam_v2
from keras import backend as K
from keras.utils.vis_utils import plot_model
import pickle
import numpy as np
import tensorflow as tf
from auxiliary.load_data import load_img
import os
import matplotlib.pyplot as plt
from auxiliary.preprocessing import  preprocess
from sklearn.metrics import roc_curve, auc



class Siamase():
    def __init__(self):
        self.rows=155
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)

        self.batchsize=40
        self.epochs=10


        self.subnet=self.bulid_model()
        self.optimizer= adam_v2.Adam(learning_rate=0.0003,beta_1=0.5)

        sig1=Input(shape=self.imgshape)
        sig2=Input(shape=self.imgshape)
        label=Input(shape=(1,)) # denote pairs

        feature1=self.subnet(sig1)
        feature2=self.subnet(sig2)

        dw=Lambda(self.Eucilid,name='distance')([feature1,feature2])
        contra_loss=Lambda(self.build_loss,name='loss')([dw,label])
        self.SigNet=Model([sig1,sig2,label],[dw,contra_loss])

        loss_layer=self.SigNet.get_layer('loss').output
        self.SigNet.add_loss(loss_layer)
        self.SigNet.compile(optimizer=self.optimizer)
        plot_model(self.SigNet, to_file='siamese.png', show_shapes=True)
        self.SigNet.summary()

    def bulid_model(self):
        model=Sequential()

        model.add(Conv2D(48,kernel_size=(11,11),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv1',activation='relu'))
        #model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='pool1'))

        model.add(Conv2D(64, kernel_size=(5, 5), strides=1, padding='same',name='conv2',activation='relu'))
        #model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',name='conv3',activation='relu'))
        #model.add(Activation('relu'))

        model.add(Conv2D(96, kernel_size=(3, 3), strides=1, padding='same',name='conv4',activation='relu'))
       #model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool3'))
        model.add(Dropout(0.3))

        #model.add(Flatten())
        #model.add(Dense(1024,name='fc1'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128,name='fc2',activation='relu'))
        #model.add(Activation('relu'))

        model.summary()

        img=Input(shape=self.imgshape)
        feature=model(img)
        plot_model(model,to_file='subnet.png',show_shapes=True)
        return  Model(img,feature)

    def  Eucilid(self,args):
        feature1,feature2= args
        dw = K.sqrt(K.sum(K.square(feature1 - feature2),axis=1))  # Euclidean distance
        return  dw

    def build_loss(self,args,alpha=0.5,beta=0.5):
        dw,label=args
        hingeloss=K.maximum(1-dw,0) # maximum(1 - y_true * y_pred, 0)
        label=tf.cast(label,tf.float32) # previously setting label int8 type tensor,can't do element wise with dw
        contrastive_loss=K.sum(label*alpha*K.square(dw)+(1-label)*beta*K.square(hingeloss))/self.batchsize
        return contrastive_loss

    def train(self,dataset,weights=''):
        save_dir = '../../NetWeights/Siamase_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.SigNet.load_weights(filepath)
        else:
            filepath = os.path.join(save_dir, 'siamase_weight.h5')
            dataset = dataset.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            i=0
            times = 0
            doc=[]
            pre_loss=[]
            min_loss=100 # manually set
            for batch in dataset:
                loss = self.SigNet.train_on_batch(batch)
                doc.append(loss)
                print("%d loss: %f " % (i, loss))
                if(i%10==0):
                    times += 1
                    if(early_stop(10,loss,pre_loss,threshold=0.5)):
                        break
                if(loss<min_loss and times>10): # to avoid frequently saving in the early stage
                    print('save')
                    self.SigNet.save_weights(filepath)
                    min_loss=loss
                i+=1
            return doc


def early_stop(stop_round,loss,pre_loss,threshold=0.002):
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


def judegement(T,N,d):
    '''

    :param TP: (ndarry) pos_pairs' distance
    :param NP: (ndarry) neg_pairs' distance
    :param d: (float32)
    :return: accuarcy
    '''
    TPR=(T[:,0]<d).sum()/T.shape[0]
    TNR=(N[:,0]>d).sum()/T.shape[0]
    acc=1/2*(TPR+TNR)
    return acc


def curve_eval(label,result):
    fpr, tpr, thresholds = roc_curve(label,result, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
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
    SigNet=Siamase()
    with open('./pair_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)
    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1], train_ind[:, 2].astype(np.int8)))
    #dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1],train_ind[:, 2],train_ind[:, 3], train_ind[:, 4].astype(np.int8)))

    image = dataset.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8, tf.uint8, tf.int8]))
    #image = dataset.map(
        #lambda x1,x2,y1,y2,z: tf.py_function(func=load_img, inp=[x1,x2,y1,y2,z], Tout=[tf.uint8,tf.uint8,tf.int8]))
    image=image.shuffle(500)
    doc=SigNet.train(image,'siamase_weight.h5')

    mode='train'

    assert  mode=='train' or mode== 'test', 'the programmer can only execute in training or testing model'

    if mode=='train':
        if(doc): # 进行了训练就画图，直接读档模型不画图
            plt.plot(doc)
            plt.title('contrastive_loss curve')
            plt.xlabel('times')
            plt.ylabel('contrastive loss')
        start=time.time()
        result=[]
        label=[]
        cost=[]
        for b in image.batch(40):
            result.append(SigNet.SigNet.predict_on_batch(b)[0])
            cost.append(SigNet.SigNet.predict_on_batch(b)[1])
            label.append(b[2].numpy())
        temp=np.array([])
        for i in result:
            temp=np.concatenate([temp,i])
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        cost=np.array(cost).reshape(-1,1)
        curve_eval(label,result)
        temp_result=np.vstack([result,label])

         # ensure threshold according to lecture
        T=temp_result[temp_result[:,1]==1,:]
        N=temp_result[temp_result[:,1]==0,:]
        print(N.mean(axis=0))
        print(T.mean(axis=0))
        d=np.arange(0,1,0.01)
        acc=[]
        for i in d:
            acc.append(judegement(T,N,i))
        acc=np.array(acc)
        threshold=d[acc.argmax()]
        end=time.time()
        print("time cost : %f"%(end-start))

    else:
        threshold=0 # must implement training stage to speacify threshold
        with open('../../pair_ind/sigcomp_ind/test_index.pkl', 'rb') as test_index_file:
            test_ind = pickle.load(test_index_file)
        test_ind = np.array(test_ind)
        test_set= tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1], test_ind[:, 2].astype(np.int8)))
        test_image = test_set.map(
            lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z,1100,2900], Tout=[tf.uint8, tf.uint8, tf.int8]))
        result=[]
        label=[]
        cost=[]
        for b in test_image.batch(40):
            result.append(SigNet.SigNet.predict_on_batch(b)[0])
            cost.append(SigNet.SigNet.predict_on_batch(b)[1])
            label.append(b[2].numpy())
        temp=np.array([])
        for i in result:
            temp=np.concatenate([temp,i])
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        curve_eval(label,result)
        cost=np.array(cost).reshape(-1,1)

        temp_result=np.vstack([result,label])
        T=temp_result[temp_result[:,1]==1,:]
        N=temp_result[temp_result[:,1]==0,:]
        print(N.mean(axis=0))
        print(T.mean(axis=0))
        acc=judegement(T,N,threshold)

def USVM():
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

    final_result=[]
    for user in range(1,16): # 测试15个用户吧
        train_data=[]
        # 15个正样本，15个random forgies做负样本训练SVM
        for j in range(1,16): # 这里不用随机应该没啥问题，让代码好看点吧
            train_data.append([org_path%(user,j),1])
        for j in range(user,user+15):  # 这里不用随机应该没啥问题，让代码好看点吧
            train_data.append([org_path%(j,2),0])
        train_imgs=[]
        for i in train_data:
            train_img = tf.io.read_file(i[0], 'rb')  # 读取图片
            train_img = tf.image.decode_png(train_img, channels=3)
            train_img = tf.image.rgb_to_grayscale(train_img)
            train_img=preprocess(train_img,820,980)
            train_imgs.append([train_img])

        train_imgs=tf.data.Dataset.from_tensor_slices(train_imgs)
        train_vecs=SigNet.subnet.predict(train_imgs)
        label=np.concatenate([np.ones(15),np.zeros(15)])
        label=label.astype(np.int32)  # opencv里SVM要求label需要为int32类型
        svm=cv.ml.SVM_create()
        svm.setKernel(cv.ml.SVM_RBF)
        svm.setType(cv.ml.SVM_C_SVC)
        result=svm.train(train_vecs,cv.ml.ROW_SAMPLE,label)

        test_data=[]
        label=np.zeros(27)
        label[0:9]=1
        # 测试时使用9个正样本，18个负样本（9个random forgies，9个skilled forgies）
        for j in range(16,25):
            test_data.append([org_path%(user,j),1])
        for j in range(user+16,user+25):
            test_data.append([org_path%(j,2),0])
        for j in range(1,10):
            test_data.append([forg_path%(user,j),0])

        test_imgs=[]
        for i in test_data:
            test_img = tf.io.read_file(i[0], 'rb')  # 读取图片
            test_img = tf.image.decode_png(test_img, channels=3)
            test_img = tf.image.rgb_to_grayscale(test_img)
            test_img=preprocess(test_img,820,980)
            test_imgs.append([test_img])
        test_imgs=tf.data.Dataset.from_tensor_slices(test_imgs)
        test_vecs=SigNet.subnet.predict(test_imgs)
        result=svm.predict(test_vecs)[1]
        temp=np.hstack([result,label.reshape(-1,1)])
        final_result.append(temp)