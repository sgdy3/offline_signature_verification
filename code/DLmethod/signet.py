import pandas as pd
from keras.layers import  Conv2D,BatchNormalization,Dense,Input,MaxPool2D,Activation,Flatten
from keras.models import Sequential,Model
import numpy as np
from tensorflow.python.keras.optimizer_v1 import SGD
import os
import tensorflow as tf
from auxiliary.preprocessing import preprocess
import keras.utils.np_utils


class signet():
    def __init__(self,num_class):
        self.rows=155
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)
        self.user_dim=num_class

        self.batchsize=32
        self.epochs=30
        self.optimizer=SGD(learning_rate=5e-4,momentum=0.9,nesterov=True,decay=5e-4)

        self.backbone=self.base_line()
        input=Input(shape=self.imgshape)
        x=self.backbone(input)
        output=Dense(self.user_dim,activation='softmax')(x)
        self.signet=Model(input,output)
        self.signet.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics='accuracy')
        self.signet.summary()


    def base_line(self):
        model=Sequential()

        model.add(Conv2D(64,kernel_size=5,strides=5,input_shape=self.imgshape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Conv2D(96,kernel_size=5,strides=2,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Conv2D(128,kernel_size=3,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(128,kernel_size=3,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.summary()
        input=Input(shape=self.imgshape)
        output=model(input)

        return Model(input,output)

    def train(self,data,weights=''):
        save_dir = '../../NetWeights/Signet_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.signet.load_weights(filepath)
            doc=None
        else:
            filepath = os.path.join(save_dir, 'signet.h5')
            train_img=data.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            time=0
            doc=[]
            for batch in train_img:
                train_label=keras.utils.np_utils.to_categorical(batch[1],num_classes=self.user_dim)
                loss=self.signet.train_on_batch(x=batch[0],y=train_label)
                doc.append(loss)
                print("round %d=> loss:%f, acc:%f%% " % (time,loss[0],loss[1]*100))
                time+=1
            self.signet.save_weights(filepath)
        return doc

def img_preprocess(file_name1,m_lab,f_lab,mod='train',ext_h=820,ext_w=890):
    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=preprocess(img1,ext_h,ext_w)
    return img1,m_lab,f_lab

def path_extra(user_ord,class_nums):
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    org_num = 24
    label_dict=dict(zip(user_ord,range(class_nums)))  # 原用户标签映射到用户数量range之内，以进行独热编码
    file_path=[]
    for user in user_ord:
        for i in range(1,org_num+1):
            positive = [org_path % (user, i),label_dict[user],1]
            negative = [forg_path % (user, i),label_dict[user],0]
            file_path.append(positive)
            file_path.append(negative)
    return np.array(file_path)

def temp_process(img, ext_h, ext_w,dst_h=155,dst_w=220):
    img=np.expand_dims(img,axis=-1)
    h,w,_=img.shape
    scale=min(ext_h / h, ext_w / w)
    nh=int(scale*h)
    nw=int(scale*w)
    img=tf.image.resize(img,[nh,nw])
    pad_row1=int((ext_h - img.shape[0]) / 2)
    pad_row2= (ext_h - img.shape[0]) - pad_row1
    pad_col1=int((ext_w - img.shape[1]) / 2)
    pad_col2= (ext_w - img.shape[1]) - pad_col1
    img=tf.pad(img,[[pad_row1,pad_row2],[pad_col1,pad_col2],[0,0]],constant_values=255)
    img=tf.image.resize(img,[dst_h,dst_w])
    #img=otsu(img.numpy())
    img=tf.convert_to_tensor(img)
    return img


if __name__=="__main__":
    net=signet(num_class=45)
    # 用户无关训练阶段
    np.random.seed(3)
    train_user_order=np.random.choice(range(1,56),45,replace=False)
    train_ind=path_extra(train_user_order,net.user_dim)
    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1].astype(np.int8),train_ind[:, 2].astype(np.int8)))
    train_image=dataset.map(lambda x, y, z: tf.py_function(func=img_preprocess, inp=[x, y, z], Tout=[tf.uint8, tf.int8, tf.int8]))
    doc=net.train(train_image,'signet.h5')

    # 用户相关判决阶段
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

    test_ind=[]
    test_user_order=np.arange(1,56)[~np.isin(np.arange(1,56),train_user_order)] # 得到测试集用户
    for user in test_user_order:
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

    from sklearn.manifold import TSNE
    tsne=TSNE(init='pca')
    feature_embedded=tsne.fit_transform(test_vec)
    feature_embedded=pd.DataFrame(feature_embedded,columns=['dim_1','dim_2'])
    label=pd.DataFrame(test_label,columns=['User','Authenticity'])
    import seaborn as sns
    sns.scatterplot(data=pd.concat([feature_embedded,label],axis=1),x='dim_1',y='dim_2',hue='User',style='Authenticity',palette='deep')