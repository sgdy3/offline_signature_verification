import tensorflow as tf
from keras.layers import Conv2D,Dense,Input,BatchNormalization
from keras.layers import Softmax,GlobalAveragePooling2D,Subtract,AvgPool2D
from keras.models import Model
import keras.utils.np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import adam_v2
import numpy as np
import cv2 as cv
import pickle
import time
import matplotlib.pyplot as plt
from auxiliary.preprocessing import hafemann_preprocess
from sklearn.metrics import roc_curve, auc
import os


class Multi_channel():
    def __init__(self):
        self.rows=150
        self.cols=220
        self.channles=2
        self.imgshape = (self.rows, self.cols, self.channles)

        self.batchsize=32
        self.epochs=1


        self.f_net1=self.feature_net()
        self.f_net1._name='contours_feature'
        self.f_net2=self.feature_net()
        self.f_net2._name='edge_feature'
        self.f_net3=self.feature_net()
        self.f_net3._name='inverse_feature'
        self.discriminator=self.judging_net()
        self.discriminator_name='discriminator'
        self.optimizer= adam_v2.Adam(learning_rate=0.0003,beta_1=0.5)

        sig1=Input(shape=self.imgshape,name='contours map1')
        sig2=Input(shape=self.imgshape,name='edge map1')
        sig3=Input(shape=self.imgshape,name='inverse map1')
        sig4=Input(shape=self.imgshape,name='contours map2')
        sig5=Input(shape=self.imgshape,name='edge map2')
        sig6=Input(shape=self.imgshape,name='inverse map2')
        label=Input(shape=(1,)) # denote pairs

        contours_feature1=self.f_net1(sig1)
        edge_feature1=self.f_net2(sig2)
        contrast_feature1=self.f_net3(sig3)

        contours_feature2=self.f_net1(sig4)
        edge_feature2=self.f_net2(sig5)
        contrast_feature2=self.f_net3(sig6)

        multi_channel_input=tf.concat([contours_feature1,contours_feature2,edge_feature1,edge_feature2,contrast_feature1,contrast_feature2],axis=-1)

        mixed_feature=self.discriminator(multi_channel_input)
        logit1=Dense(2,activation='relu',name='logit1')(mixed_feature)
        logit2=Dense(2,activation='relu',name='logit2')(mixed_feature)
        diff_logit=Subtract(name='diff_logit')([logit1,logit2])
        class_prob=Softmax(name='prob')(diff_logit)

        self.Mnet=Model([sig1,sig2,sig3,sig4,sig5,sig6,label],[class_prob])
        self.Mnet.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics='accuracy')
        self.Mnet.summary()
        plot_model(self.Mnet, to_file='MNet.png', show_shapes=True)

    def feature_net(self):

        input=Input(shape=(self.rows,self.cols,self.channles),name='multi_input')
        outs=Conv2D(64,kernel_size=3,strides=2,padding='same',activation='relu')(input)
        outs=BatchNormalization(momentum=0.9)(outs)
        outs=Conv2D(32,kernel_size=3,strides=2,padding='same',activation='relu')(outs)
        outs=BatchNormalization(momentum=0.9)(outs)
        outs=self.incep_modual()(outs)
        outs=Conv2D(1,kernel_size=3,strides=1,padding='same',activation='relu')(outs)
        f_net=Model(input,outs,name='feature_net')
        f_net.summary()
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
        incep.summary()
        return incep

    def judging_net(self):
        input=Input(shape=(38,55,6))
        outs=Conv2D(64,kernel_size=3,strides=1,padding='same')(input)
        outs=Conv2D(32,kernel_size=3,strides=1,padding='same',activation='relu')(outs)
        outs=self.incep_modual()(outs)
        outs=GlobalAveragePooling2D()(outs)
        outs=Dense(128,activation='relu')(outs)
        judging=Model(input,outs,name='judging_net')
        judging.summary()
        return judging

    def train(self,img_set,weights=''):
        save_dir = '../../NetWeights/Mnet_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.Mnet.load_weights(filepath)
            return None
        else:
            filepath = os.path.join(save_dir, 'Mnet_weight.h5')
            img_set=img_set.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            times=1
            acc_loss=[]
            doc=[]
            for batch in img_set:
                # if((times>50) and (times%10==0)):
                #     lr=self.Mnet.optimizer.lr
                #     self.Mnet.optimizer.lr=0.99*lr
                label=keras.utils.np_utils.to_categorical(batch[6])
                loss,metric=self.Mnet.train_on_batch(batch,label)
                doc.append([loss,metric])
                print("round %d=> loss:%f, acc:%f%% " % (times,loss,metric*100))
                if(early_stop(20,loss,acc_loss)):
                    print("training complete")
                    print("Final result=> loss:%f, acc:%f%% " % (loss,metric))
                    self.Mnet.save_weights(filepath)
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
    #img1=preprocess(img1,ext_h,ext_w)
    img1=hafemann_preprocess(img1.numpy(),ext_h,ext_w)
    img1=np.expand_dims(img1,axis=2)
    surf=cv.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img1,None)
    temp_img=np.squeeze(img1)
    pt=[i.pt for i in kp]
    pt=np.array(pt)
    loc=np.zeros((pt.shape[0],4))
    loc[:,0]=pt[:,1]-2
    loc[:,1]=pt[:,0]-2
    loc[:,2]=pt[:,1]+2
    loc[:,3]=pt[:,0]+2 # 囊括特征点周围3*3的领域,用检测出的size的话太大了
    loc=loc.astype(int)
    contours_map=np.zeros(temp_img.shape)
    if len(kp)>20:
        for i in range(20):
            pos=loc[i]
            contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    else:
        for i in range(len(kp)):
            pos=loc[i]
            contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    contours_map1=contours_map.astype(np.uint8)
    contours_map1=np.expand_dims(contours_map1,axis=2)
    contours_map1=tf.concat([contours_map1,tf.convert_to_tensor(img1)],axis=-1)

    edge_map1=cv.Canny(img1,50,150)
    edge_map1=np.expand_dims(edge_map1,axis=2)
    edge_map1=tf.concat([edge_map1,tf.convert_to_tensor(img1)],axis=-1)

    inverse_map1=cv.bitwise_not(img1)
    inverse_map1=np.expand_dims(inverse_map1,axis=2)
    inverse_map1=tf.concat([inverse_map1,tf.convert_to_tensor(img1)],axis=-1)

    img2 = tf.io.read_file(file_name2, 'rb')  # 读取图片
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.rgb_to_grayscale(img2)
    #img2=preprocess(img2,ext_h,ext_w)
    img2=hafemann_preprocess(img2,ext_h,ext_w)
    img2=np.expand_dims(img2,axis=2)

    kp, des = surf.detectAndCompute(img2,None)
    temp_img=np.squeeze(img2)
    pt=[i.pt for i in kp]
    pt=np.array(pt)
    loc=np.zeros((pt.shape[0],4))
    loc[:,0]=pt[:,1]-2
    loc[:,1]=pt[:,0]-2
    loc[:,2]=pt[:,1]+2
    loc[:,3]=pt[:,0]+2 # 囊括特征点周围3*3的领域,用检测出的size的话太大了
    loc=loc.astype(int)
    contours_map=np.zeros(temp_img.shape)
    if len(kp)>20:
        for i in range(20):
            pos=loc[i]
            contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    else:
        for i in range(len(kp)):
            pos=loc[i]
            contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
    contours_map2=contours_map.astype(np.uint8)
    contours_map2=np.expand_dims(contours_map2,axis=2)
    contours_map2=tf.concat([contours_map2,tf.convert_to_tensor(img2)],axis=-1)

    edge_map2=cv.Canny(img2,50,150)
    edge_map2=np.expand_dims(edge_map2,axis=2)
    edge_map2=tf.concat([edge_map2,tf.convert_to_tensor(img2)],axis=-1)

    inverse_map2=cv.bitwise_not(img2)
    inverse_map2=np.expand_dims(inverse_map2,axis=2)
    inverse_map2=tf.concat([inverse_map2,tf.convert_to_tensor(img2)],axis=-1)

    return contours_map1,edge_map1,inverse_map1,contours_map2,edge_map2,inverse_map2,label

def test(data,model):
    start=time.time()
    result=[]
    label=[]
    cost=[]
    for b in data.batch(32):
        result.append(model.Mnet.predict_on_batch(b))
        label.append(b[6].numpy()) # 第7个元素是标签
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

    fpr, tpr, thresholds = roc_curve(label,result[:,1], pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result[:,1].copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size

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
    plt.title('ROC on training set')
    plt.legend(loc="lower right")
    plt.show()

    return area,EER,acc

if __name__=='__main__':
    Mnet=Multi_channel()
    with open('../../pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)
    # 预先对ind打乱实现shuffle，避免训练过程中shuffle耗时过长
    train_ind=train_ind[np.random.permutation(train_ind.shape[0]),:]

    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1], train_ind[:, 2].astype(np.int8)))


    image = dataset.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.uint8,tf.uint8,tf.uint8,tf.uint8,tf.uint8,tf.int8]))
    start=time.time()
    doc=Mnet.train(image)
    end=time.time()

    with open('../../pair_ind/cedar_ind/test_index.pkl', 'rb') as test_index_file:
        test_ind = pickle.load(test_index_file)
    test_ind = np.array(test_ind)
    test_set= tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1], test_ind[:, 2].astype(np.int8)))
    test_image = test_set.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.uint8,tf.uint8,tf.uint8,tf.uint8,tf.uint8,tf.int8]))

    start=time.time()
    result=[]
    label=[]
    cost=[]
    for b in test_image.batch(32):
        result.append(Mnet.Mnet.predict_on_batch(b))
        label.append(b[6].numpy())
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

    fpr, tpr, thresholds = roc_curve(label,result[:,1], pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result[:,1].copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size

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
    plt.title('ROC on training set')
    plt.legend(loc="lower right")
    plt.show()






