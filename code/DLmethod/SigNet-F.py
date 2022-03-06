from keras.layers import Conv2D,MaxPool2D,Dense,Input,BatchNormalization,Lambda,Activation,Flatten
from keras.models import Sequential,Model
from keras.losses import categorical_crossentropy,binary_crossentropy
import keras.backend as K
from tensorflow.python.keras.optimizer_v1 import SGD
import pickle
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import re
from auxiliary.preprocessing import preprocess
import keras.utils.np_utils
import os
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class SigNet_F():
    def __init__(self):
        self.rows=155
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)
        self.user_dim=20

        self.batchsize=32
        self.epochs=10

        self.optimizer=SGD(learning_rate=1e-3,momentum=0.9,nesterov=True,decay=5e-4)

        self.backbone=self.base_line()
        sig=Input(shape=self.imgshape)
        m_label=Input(shape=(self.user_dim,))
        f_label=Input(shape=(1,))

        feature=self.backbone(sig)
        pred_m=Dense(self.user_dim)(feature)
        pred_f=Dense(1)(feature)
        mixed_loss=Lambda(self.combine_loss,name='loss')([m_label,pred_m,f_label,pred_f])
        self.signet_f=Model([sig,m_label,f_label],[pred_m,pred_f,mixed_loss])

        loss_layer=self.signet_f.get_layer('loss').output
        self.signet_f.add_loss(loss_layer)
        self.signet_f.compile(optimizer=self.optimizer)
        plot_model(self.signet_f, to_file='signet_f.png', show_shapes=True)
        self.signet_f.summary()


    def combine_loss(self,args,alpha=0.99):
        m_label,pred_m,f_label,pred_f=args
        cat_los=categorical_crossentropy(m_label,pred_m)
        b_los=binary_crossentropy(f_label,pred_f)
        return K.mean((1-alpha)*cat_los+alpha*b_los)

    def base_line(self):
        seq=Sequential()

        # 155*220->37*53
        seq.add(Conv2D(32,kernel_size=11,strides=4,input_shape=self.imgshape))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 37*53->18*26
        seq.add(MaxPool2D(pool_size=3,strides=2))

        # 18*26->8*12
        seq.add(Conv2D(64,kernel_size=5,strides=1,padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->8*12
        seq.add(MaxPool2D(pool_size=3,strides=2))

        # 8*12->8*12
        seq.add(Conv2D(64,kernel_size=3,strides=1,padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->8*12
        seq.add(Conv2D(96,kernel_size=3,strides=1,padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->8*12
        seq.add(Conv2D(96,kernel_size=5,strides=1,padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        # 8*12->3*5
        seq.add(MaxPool2D(pool_size=3,strides=2))

        # 3*5->2048*1
        seq.add(Flatten())
        seq.add(Dense(128))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))

        seq.summary()

        # user_dim->binary
        img=Input(shape=self.imgshape)
        feature=seq(img)

        return Model(img,feature)

    def train(self,data,weights=''):
        save_dir = '../../NetWeights/Signet_f_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.signet_f.load_weights(filepath)
            doc=None
        else:
            filepath = os.path.join(save_dir, 'signet_f.h5')
            train_img=data.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            time=0
            doc=[]
            for batch in train_img:
                loss=self.signet_f.train_on_batch(batch)
                doc.append(loss)
                print("%d round: loss %f"%(time,loss))
                time+=1
            self.signet_f.save_weights(filepath)
        return doc

def img_preprocess(file_name1,m_lab,f_lab,mod='train',ext_h=820,ext_w=890):
    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=preprocess(img1,ext_h,ext_w)
    if mod=='train':
        m_lab= keras.utils.np_utils.to_categorical(m_lab,20)
    elif mod=='test':
        m_lab=m_lab
    else:
        raise Exception("mod doesn't exist")
    return img1,m_lab,f_lab

def path_extra(user_ord):
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    org_num = 24
    label_dict=dict(zip(user_ord,range(user_ord.shape[0])))  # 原用户标签映射到用户数量range之内
    file_path=[]
    for user in user_ord:
        for i in range(1,org_num+1):
            positive = [org_path % (user, i),label_dict[user],1]
            negative = [forg_path % (user, i),label_dict[user],0]
            file_path.append(positive)
            file_path.append(negative)
    return np.array(file_path)



if __name__=="__main__":
    net=SigNet_F()
    with open('../../pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)

    # 为了保证对比合理，SigNet-F和其他方式使用相同的用户训练
    # 但由于其他方案是pair输入，为了使得正负样本对数目相当，没有使用所用的负样本
    # 所以先提取出用户再使用其所有的样本
    user_order=[]
    for i in range(train_ind.shape[0]):
        user_order.append(int(re.search('(?<=_)\d+(?=_)',train_ind[i][0]).group())) # 提取测试图片的用户编号
    user_order=np.unique(user_order)
    train_ind=path_extra(user_order)
    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1].astype(np.int8),train_ind[:, 2].astype(np.int8)))
    image=dataset.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z], Tout=[tf.uint8, tf.int8, tf.int8]))
    doc=net.train(image,'signet_f.h5')



    # 用户相关判决阶段
    # 这里用testfile里的就过于麻烦了， 每个用户的在testfile里的样本数目都不一定相同
    # 不如直接重新训练一个分类器，每个用户使用12个真实样本作为正例，训练集中用户的正式样本作为负例
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    train_ind=train_ind[train_ind[:,2].astype(int)==1]
    neg_vecs=[]
    neg_database=tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1].astype(np.int8),train_ind[:, 2].astype(np.int8)))
    neg_database=neg_database.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z], Tout=[tf.uint8, tf.int8, tf.int8]))
    for batch in neg_database.batch(32):
        neg_vecs.append(net.backbone.predict_on_batch(batch[0])) # 获得训练集中用户真实签名的特征向量
    neg_vecs=np.vstack(neg_vecs)

    test_ind=[]
    user_order=np.arange(1,56)[~np.isin(np.arange(1,56),user_order)] # 得到测试集用户
    for user in user_order:
        for i in range(1,25):
            pos=[org_path%(user,i),user,1]
            neg=[forg_path%(user,i),user,0]
            test_ind.append(pos)
            test_ind.append(neg)
    test_ind=np.array(test_ind)
    test_vec=[]
    test_label=[]
    test_sig=tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1].astype(np.int8),test_ind[:, 2].astype(np.int8)))
    test_sig=test_sig.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z,'test'], Tout=[tf.uint8, tf.int8, tf.int8]))
    for batch in test_sig.batch(32):
        test_vec.append(net.backbone.predict_on_batch(batch[0]))
        test_label.append(np.vstack([batch[1],batch[2]]))
    test_vec=np.vstack(test_vec)
    test_label=np.hstack(test_label).T

    result=[]
    for user in user_order:
        user_ind=np.where(test_label[:,0]==user)[0] # test库中用户记录
        user_pos_ind=np.where((test_label[:,0]==user) & (test_label[:,1]==1))[0] # test库中用户真实样本记录
        user_train_ind=np.random.choice(user_pos_ind,12,replace=False) # 随机采样24个真实签名中的12个做正样本
        user_test_ind=user_ind[~np.isin(user_ind,user_train_ind)]

        skew = neg_vecs.shape[0] / user_pos_ind.shape[0]
        svm_input=np.vstack([neg_vecs,test_vec[user_pos_ind,:]])
        svm_label=np.concatenate([np.zeros(neg_vecs.shape[0]),np.ones(user_pos_ind.shape[0])])
        svm=sklearn.svm.SVC(class_weight={1:skew},gamma=0.0048,probability=True)
        svm_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                               ('classifier', svm)])
        svm_with_scaler.fit(svm_input,svm_label)
        hyper_dist=svm_with_scaler.decision_function(test_vec[user_test_ind,:])
        result.append(np.vstack([hyper_dist,test_label[user_test_ind,1]]))

    result=np.hstack(result).T

    fpr, tpr, thresholds = roc_curve(result[:,1],result[:,0], pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result[:,1].copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==result[:,1]).sum()/result.shape[0]

    area = auc(fpr, tpr)
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












