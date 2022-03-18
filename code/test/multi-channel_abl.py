import tensorflow as tf
from keras.layers import Conv2D,Dense, Input,BatchNormalization
from keras.layers import Softmax,GlobalAveragePooling2D,Subtract,AvgPool2D
from keras.models import Model
import keras.utils.np_utils
from keras.optimizers import adam_v2
import numpy as np
import cv2 as cv
import pickle
import time
import matplotlib.pyplot as plt
from auxiliary.preprocessing import preprocess,hafemann_preprocess
from sklearn.metrics import roc_curve, auc
import os


class Multi_channel():
    def __init__(self):
        self.rows=150
        self.cols=220
        self.channles=1
        self.mixed_channels=2
        self.imgshape = (self.rows, self.cols, self.channles)

        self.batchsize=32
        self.epochs=1


        self.f_net1=self.feature_net()
        #self.f_net2=self.feature_net()
        #self.f_net3=self.feature_net()
        self.discriminator=self.judging_net()
        self.optimizer= adam_v2.Adam(learning_rate=0.0003)

        sig1=Input(shape=self.imgshape)
        sig2=Input(shape=self.imgshape)
        sig3=Input(shape=self.imgshape)
        sig4=Input(shape=self.imgshape)
        sig5=Input(shape=self.imgshape)
        sig6=Input(shape=self.imgshape)
        label=Input(shape=(1,)) # denote pairs

        contours_feature1=self.f_net1(sig1)
        #edge_feature1=self.f_net2(sig2)
        #contrast_feature1=self.f_net3(sig3)

        contours_feature2=self.f_net1(sig4)
        #edge_feature2=self.f_net2(sig5)
        #contrast_feature2=self.f_net3(sig6)

        #multi_channel_input=tf.concat([contours_feature1,contours_feature2,edge_feature1,edge_feature2,contrast_feature1,contrast_feature2],axis=-1)
        multi_channel_input=tf.concat([contours_feature1,contours_feature2],axis=-1)

        mixed_feature=self.discriminator(multi_channel_input)
        logit1=Dense(2,activation='relu')(mixed_feature)
        logit2=Dense(2,activation='relu')(mixed_feature)
        diff_logit=Subtract(name='sub')([logit1,logit2])
        class_prob=Softmax()(diff_logit)

        self.Mnet=Model([sig1,sig4,label],[class_prob])
        self.Mnet.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics='accuracy',)
        self.Mnet.summary()

    def feature_net(self):

        input=Input(shape=(self.rows,self.cols,self.channles),name='multi_input')
        outs=Conv2D(64,kernel_size=3,strides=2,padding='same',activation='relu')(input)
        outs=BatchNormalization(momentum=0.9)(outs)
        outs=Conv2D(32,kernel_size=3,strides=2,padding='same',activation='relu')(outs)
        outs=BatchNormalization(momentum=0.9)(outs)
        outs=self.incep_modual()(outs)
        outs=Conv2D(1,kernel_size=3,strides=1,padding='same',activation='relu')(outs)
        f_net=Model(input,outs)
        return f_net

    def incep_modual(self):
        input=Input(shape=(38,55,32),name='inception_input')
        x1=Conv2D(64,kernel_size=1,strides=1,padding='same',activation='relu')(input)
        x1=BatchNormalization(momentum=0.9)(x1)

        x2=Conv2D(48,kernel_size=1,strides=1,padding='same',activation='relu')(input)
        x2=BatchNormalization(momentum=0.9)(x2)
        x2=Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(x2)
        x2=BatchNormalization(momentum=0.9)(x2)

        x3=Conv2D(48,kernel_size=1,strides=1,padding='same',activation='relu')(input)
        x3=BatchNormalization(momentum=0.9)(x3)
        x3=Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(x3)
        x3=BatchNormalization(momentum=0.9)(x3)
        x3=Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(x3)
        x3=BatchNormalization(momentum=0.9)(x3)

        x4=AvgPool2D(pool_size=3,strides=1,padding='same')(input)
        x4=Conv2D(32,kernel_size=1,strides=1,padding='same',activation='relu')(x4)
        x4=BatchNormalization(momentum=0.9)(x4)
        output=tf.concat([x1,x2,x3,x4],axis=3)

        incep=Model(input,output,name='inception')
        return incep

    def judging_net(self):
        input=Input(shape=(38,55,self.mixed_channels))
        outs=Conv2D(64,kernel_size=3,strides=1,padding='same')(input)
        outs=Conv2D(32,kernel_size=3,strides=1,padding='same',activation='relu')(outs)
        outs=self.incep_modual()(outs)
        outs=GlobalAveragePooling2D()(outs)
        outs=Dense(128,activation='relu')(outs)

        judging=Model(input,outs)
        return judging

    def train(self,img_set,weights=''):
        img_set=img_set.shuffle(100).batch(self.batchsize).repeat(self.epochs)
        times=1
        acc_loss=[]
        doc=[]
        for batch in img_set:
            # if((times>50) and (times%10==0)):
            #     lr=self.Mnet.optimizer.lr
            #     self.Mnet.optimizer.lr=0.99*lr
            label=keras.utils.np_utils.to_categorical(batch[2])
            loss,metric=self.Mnet.train_on_batch(batch,label)
            doc.append([loss,metric])
            print("round %d=> loss:%f, acc:%f%% " % (times,loss,metric*100))
            if(early_stop(20,loss,acc_loss)):
                print("training complete")
                print("Final result=> loss:%f, acc:%f%% " % (loss,metric))
                break
            if (times>200):
                print("enough rounds!!")
                break
            times+=1
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


def load_img(file_name1,file_name2,label,ext_h=820,ext_w=890):
    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=hafemann_preprocess(img1.numpy(),ext_h,ext_w)
    img1=np.expand_dims(img1,-1)
    # img1=preprocess(img1,ext_h,ext_w)

    # surf=cv.xfeatures2d.SURF_create(400)
    # kp, des = surf.detectAndCompute(img1,None)
    # temp_img=np.squeeze(img1).copy()
    # pt=[i.pt for i in kp]
    # pt=np.array(pt)
    # loc=np.zeros((pt.shape[0],4))
    # if pt.shape[0]!=0:
    #     # 存在检测不到角点的情况
    #     loc[:,0]=pt[:,1]-2
    #     loc[:,1]=pt[:,0]-2
    #     loc[:,2]=pt[:,1]+2
    #     loc[:,3]=pt[:,0]+2 # 囊括特征点周围3*3的领域,用检测出的size的话太大了
    # loc=loc.astype(int)
    # contours_map=np.zeros(temp_img.shape)
    # if len(kp)>20:
    #     for i in range(20):
    #         pos=loc[i]
    #         contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    # else:
    #     for i in range(len(kp)):
    #         pos=loc[i]
    #         contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    # contours_map1=contours_map.astype(np.uint8)
    # contours_map1=np.expand_dims(contours_map1,axis=2)
    # contours_map1=tf.concat([contours_map1,img1],axis=-1)

    # edge_map1=cv.Canny(img1,50,150)
    # inter_img=np.squeeze(img1).copy()  # Warninig:np.squeeze为浅拷贝
    # inter_img[edge_map1==0]=0
    # edge_map1=np.expand_dims(inter_img,axis=2)
    # edge_map1=tf.concat([edge_map1,img1],axis=-1)

    # inverse_map1=cv.bitwise_not(img1)
    # inverse_map1=np.expand_dims(inverse_map1,axis=2)
    # inverse_map1=tf.concat([inverse_map1,img1],axis=-1)

    # cross_k=cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    # morph_map1=cv.morphologyEx(img1,cv.MORPH_OPEN,cross_k)
    # morph_map1=np.expand_dims(morph_map1,axis=2)
    # morph_map1=tf.concat([morph_map1,img1],axis=-1)


    img2 = tf.io.read_file(file_name2, 'rb')  # 读取图片
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.rgb_to_grayscale(img2)
    img2=hafemann_preprocess(img2.numpy(),ext_h,ext_w)
    img2=np.expand_dims(img2,-1)
    # img2=preprocess(img2,ext_h,ext_w)p

    # kp, des = surf.detectAndCompute(img2,None)
    # temp_img=np.squeeze(img2).copy()
    # pt=[i.pt for i in kp]
    # pt=np.array(pt)
    # loc=np.zeros((pt.shape[0],4))
    # if pt.shape[0]!=0:
    #     # 存在检测不到角点的情况
    #     loc[:,0]=pt[:,1]-2
    #     loc[:,1]=pt[:,0]-2
    #     loc[:,2]=pt[:,1]+2
    #     loc[:,3]=pt[:,0]+2 # 囊括特征点周围3*3的领域,用检测出的size的话太大了
    # loc=loc.astype(int)
    # contours_map=np.zeros(temp_img.shape)
    # if len(kp)>20:
    #     for i in range(20):
    #         pos=loc[i]
    #         contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    # else:
    #     for i in range(len(kp)):
    #         pos=loc[i]
    #         contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    # contours_map2=contours_map.astype(np.uint8)
    # contours_map2=np.expand_dims(contours_map2,axis=2)
    # contours_map2=tf.concat([contours_map2,img2],axis=-1)


    # edge_map2=cv.Canny(img2,50,150)
    # inter_img=np.squeeze(img2).copy()  # Warninig:np.squeeze为浅拷贝
    # inter_img[edge_map2==0]=0
    # edge_map2=np.expand_dims(inter_img,axis=2)
    # edge_map2=tf.concat([edge_map2,img2],axis=-1)


    # inverse_map2=cv.bitwise_not(img2)
    # inverse_map2=np.expand_dims(inverse_map2,axis=2)
    # inverse_map2=tf.concat([inverse_map2,img2],axis=-1)

    # morph_map2=cv.morphologyEx(img2,cv.MORPH_OPEN,cross_k)
    # morph_map2=np.expand_dims(morph_map2,axis=2)
    # morph_map2=tf.concat([morph_map2,img2],axis=-1)

    #return contours_map1,edge_map1,inverse_map1,contours_map2,edge_map2,inverse_map2,label

    return img1,img2,label

def draw_fig(pred,label):
    fpr, tpr, thresholds = roc_curve(label,pred, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=pred.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size
    pred_label=pred.copy()
    pred_label[pred_label>0.5]=1
    pred_label[pred_label<=0.5]=0
    acc_half=(pred_label==label).sum()/label.size

    area = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.5f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()

    return area,EER,acc,acc_half

if __name__=='__main__':
    Mnet=Multi_channel()
    with open('../../pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)
    # 预先对ind打乱实现shuffle，避免训练过程中shuffle耗时过长
    train_ind=train_ind[np.random.permutation(train_ind.shape[0]),:]
    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1], train_ind[:, 2].astype(np.int8)))

    image = dataset.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.uint8,tf.int8]))
    doc=Mnet.train(image)

    with open('../../pair_ind/cedar_ind/test_index.pkl', 'rb') as test_index_file:
        test_ind = pickle.load(test_index_file)
    test_ind = np.array(test_ind)
    test_set= tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1], test_ind[:, 2].astype(np.int8)))
    test_image = test_set.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.uint8,tf.int8]))

    start=time.time()
    result=[]
    label=[]
    cost=[]
    for b in test_image.batch(32):
        result.append(Mnet.Mnet.predict_on_batch(b))
        label.append(b[2].numpy())
    end=time.time()
    print("time cost : %f"%(end-start))

    temp=np.zeros((1,2))
    for i in result:
        temp=np.vstack([temp,i]) # 由于batch为32时不能整除，返回result的shape不都是32不能直接化为ndarray
    temp=temp[1:,:]
    result=temp.copy()
    temp=np.array([])
    for i in label:
        temp=np.concatenate([temp,i])
    label=temp.copy()

    area,EER,acc,acc_half=draw_fig(result[:,1],label)