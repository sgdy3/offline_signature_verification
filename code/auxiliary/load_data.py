# -*- coding: utf-8 -*-
# coding:unicode_escape

import numpy as np
from itertools import combinations,product
import pickle
import os
import tensorflow as tf
from preprocessing import preprocess
import re
import pandas as pd


'''
def save_pairs():
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
    sys_org_path=r'E:\material\signature\signatures\sys_org\sys_org_%d_%d.png'
    sys_forg_path=r'E:\material\signature\signatures\sys_forg\sys_forg_%d_%d.png'

    forg_author = 55  # num of writers
    org_author = 55
    forg_num = 24  # signatures of each writer
    org_num = 24
    train_size = 12000  # num of pairs for each class in train_data
    test_size = 2000

    M = 50  # num of writers for testing
    K = forg_author - M
    test_writer = np.random.randint(1, high=forg_author + 1, size=K)
    train_writer = np.arange(1, forg_author + 1)
    train_writer = train_writer[~np.isin(train_writer, test_writer)]
    np.random.shuffle(train_writer)

    pairs = np.array(list(combinations(range(1, org_num + 1), 2)))  # get pairs

    train_file_ind = []
    for i in train_writer:
        for _ in range(int(train_size / M)):
            ind = np.random.randint(0, pairs.shape[0],1)
            positive = [org_path % (i, pairs[ind, 0]),sys_org_path % (i, pairs[ind, 0]),org_path % (i, pairs[ind, 1]),sys_org_path % (i, pairs[ind, 1]),1]
            negative = [org_path % (i, pairs[ind, 0]),sys_org_path % (i, pairs[ind, 0]),forg_path % (i, pairs[ind, 1]),sys_forg_path % (i, pairs[ind, 0]),0]
            train_file_ind.append(positive)
            train_file_ind.append(negative)

    test_file_ind = []
    for i in test_writer:
        for _ in range(int(test_size / K)):
            ind = np.random.randint(0, pairs.shape[0],1)
            positive = [org_path % (i, pairs[ind, 0]),sys_org_path % (i, pairs[ind, 0]),org_path % (i, pairs[ind, 1]),sys_org_path % (i, pairs[ind, 1]),1]
            negative = [org_path % (i, pairs[ind, 0]),sys_org_path % (i, pairs[ind, 0]),forg_path % (i, pairs[ind, 1]),sys_forg_path % (i, pairs[ind, 0]),0]
            test_file_ind.append(positive)
            test_file_ind.append(negative)

    save_dir = 'pair_ind'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open('./pair_ind/train_index.pkl', 'wb') as train_index_file:
        pickle.dump(train_file_ind, train_index_file)

    with open('./pair_ind/test_index.pkl', 'wb') as test_index_file:
        pickle.dump(test_file_ind, test_index_file)
    return train_file_ind,test_file_ind


# 加载图片，带vae特征扩充版本
def load_img(file_name1,file_name2,file_name3,file_name4,label):
    img1 = tf.io.read_file(file_name1,'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1= tf.image.rgb_to_grayscale(img1)
    img1= tf.image.resize(img1, [155, 220])  # the dataype will change to float32 because of inteplotation
    img1 = tf.cast(img1,tf.uint8)
    img1=255-img1 #invert
    img1=img1.numpy() # haven't found method to appply boolean index or threshold with tf
    img1[img1>=50]=255
    img1[img1 < 50]=0
    img1=tf.convert_to_tensor(img1)

    img2=tf.io.read_file(file_name2,'rb')  # 读取图片
    img2=tf.image.decode_png(img2, channels=1)
    img2= tf.image.resize(img2, [155, 220])  # the dataype will change to float32 because of inteplotation
    img2 = tf.cast(img2,tf.uint8)
    img1=tf.concat([img1,img2],2)

    img3 = tf.io.read_file(file_name3,'rb')  # 读取图片
    img3 = tf.image.decode_jpeg(img3, channels=3)
    img3= tf.image.rgb_to_grayscale(img3)
    img3 = tf.image.resize(img3, [155, 220])  # the dataype will change to float32 because of inteplotation
    img3 = tf.cast(img3,tf.uint8)
    img3=255-img3 #invert
    img3=img3.numpy() # haven't found method to appply boolean index or threshold with tf
    img3[img3>=50]=255
    img3[img3 < 50]=0
    img3=tf.convert_to_tensor(img3)

    img4=tf.io.read_file(file_name4,'rb')  # 读取图片
    img4=tf.image.decode_png(img4, channels=1)
    img4= tf.image.resize(img4, [155, 220])  # the dataype will change to float32 because of inteplotation
    img4 = tf.cast(img4,tf.uint8)
    img3=tf.concat([img3,img4],2)

    return img1,img3,label
'''

'''
def save_pairs():
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

    forg_author = 55  # num of writers
    org_author = 55
    forg_num = 24  # signatures of each writer
    org_num = 24
    train_size = 12000  # num of pairs for each class in train_data
    test_size = 2000

    M = 50  # num of writers for testing
    K = forg_author - M
    test_writer = np.random.randint(1, high=forg_author + 1, size=K)
    train_writer = np.arange(1, forg_author + 1)
    train_writer = train_writer[~np.isin(train_writer, test_writer)]
    np.random.shuffle(train_writer)

    pairs = np.array(list(combinations(range(1, org_num + 1), 2)))  # get pairs

    train_file_ind = []
    for i in train_writer:
        for _ in range(int(train_size / M)):
            ind = np.random.randint(0, pairs.shape[0],1)
            positive = [org_path % (i, pairs[ind, 0]),org_path % (i, pairs[ind, 1]),1]
            negative = [org_path % (i, pairs[ind, 0]),forg_path % (i, pairs[ind, 1]),0]
            train_file_ind.append(positive)
            train_file_ind.append(negative)

    test_file_ind = []
    for i in test_writer:
        for _ in range(int(test_size / K)):
            ind = np.random.randint(0, pairs.shape[0],1)
            positive = [org_path % (i, pairs[ind, 0]),org_path % (i, pairs[ind, 1]),1]
            negative = [org_path % (i, pairs[ind, 0]),forg_path % (i, pairs[ind, 1]),0]
            test_file_ind.append(positive)
            test_file_ind.append(negative)

    save_dir = 'pair_ind'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open('./pair_ind/train_index.pkl', 'wb') as train_index_file:
        pickle.dump(train_file_ind, train_index_file)

    with open('./pair_ind/test_index.pkl', 'wb') as test_index_file:
        pickle.dump(test_file_ind, test_index_file)

    return train_file_ind,test_file_ind
    
'''

def save_pairs():
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

    forg_author = 55  # num of writers
    org_author = 55
    forg_num = 24  # signatures of each writer
    org_num = 24

    M = 50  # num of writers for training
    K = forg_author - M
    test_writer = np.random.choice(range(1,org_author+1),K,replace=False)
    train_writer = np.arange(1, forg_author + 1)
    train_writer = train_writer[~np.isin(train_writer, test_writer)]
    np.random.shuffle(train_writer)

    pos_pairs = np.array(list(combinations(range(1, org_num + 1), 2)))  # positive pairs, full combinations
    neg_ind=np.random.choice(range(0,org_num**2),pos_pairs.shape[0],replace=False)
    neg_pairs = np.array(list(product(range(1, forg_num + 1),range(1, forg_num + 1))))
    neg_pairs=neg_pairs[neg_ind,:]  # negative pairs,subset of full combinations
    # get pairs
    train_file_ind = []
    for i in train_writer:
        for j in range(pos_pairs.shape[0]):
            positive = [org_path % (i, pos_pairs[j, 0]),org_path % (i, pos_pairs[j, 1]),1]
            negative = [org_path % (i, neg_pairs [j,0]),forg_path % (i, neg_pairs [j,1]),0]
            train_file_ind.append(positive)
            train_file_ind.append(negative)

    train_file_ind=np.array(train_file_ind)


    test_file_ind = []
    for i in test_writer:
        for j in range(pos_pairs.shape[0]):
            positive = [org_path % (i, pos_pairs[j, 0]),org_path % (i, pos_pairs[j, 1]),1]
            negative = [org_path % (i, neg_pairs [j,0]),forg_path % (i, neg_pairs [j,1]),0]
            test_file_ind.append(positive)
            test_file_ind.append(negative)

    test_file_ind=np.array(test_file_ind)

    save_dir = '../../pair_ind'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open('../../pair_ind/cedar_ind/train_index.pkl', 'wb') as train_index_file:
        pickle.dump(train_file_ind, train_index_file)

    with open('../../pair_ind/cedar_ind/test_index.pkl', 'wb') as test_index_file:
        pickle.dump(test_file_ind, test_index_file)

    return train_file_ind,test_file_ind


def save_paris_v2():
    file_path=r'E:\material\signature\SigComp2009-training\NISDCC-offline-all-001-051-6g\NISDCC-offline-all-001-051-6g'
    dir=os.listdir(file_path)
    dir=np.array(dir)
    user=[]
    for i in dir:
        user.append(re.findall('\d{3}',i))
    user=np.array(user,dtype=int)
    user=pd.DataFrame(user,columns=['SA','PA','Seq'])

    def pairs_config(df,balance=True):
        pos_ind=df.index[np.where(df['SA']==df['PA'])[0]]
        neg_id=df.index[np.where(df['SA']!=df['PA'])[0]]
        pos_pairs=np.array(list(combinations(pos_ind,2)))
        neg_pairs = np.array(list(product(pos_ind,neg_id)))
        if balance:
            neg_pairs_ind=np.random.choice(range(neg_pairs.shape[0]),pos_pairs.shape[0],replace=False)
            neg_pairs=neg_pairs[neg_pairs_ind,:]
            ind=np.hstack([pos_pairs,neg_pairs])
            ind=pd.DataFrame(ind,columns=['pos1','pos2','pos3','neg1'])
        else:
            pos_pairs=pd.DataFrame(pos_pairs,columns=['pos1','pos2'])
            neg_pairs=pd.DataFrame(neg_pairs,columns=['pos3','neg1'])
            ind=pd.concat([pos_pairs,neg_pairs],axis=1)
        return ind
    pairs_ind=user.groupby('PA').apply(pairs_config)
    pairs_paths=[]
    for i in pairs_ind.index:
        pos_path=[os.path.join(file_path,dir[pairs_ind.loc[i,'pos1']]),os.path.join(file_path,dir[pairs_ind.loc[i,'pos2']]),1]
        neg_path=[os.path.join(file_path,dir[pairs_ind.loc[i,'pos3']]),os.path.join(file_path,dir[pairs_ind.loc[i,'neg1']]),0]
        pairs_paths.append(pos_path)
        pairs_paths.append(neg_path)
    pairs_paths=np.array(pairs_paths)
    save_dir = '../../pair_ind/sigcomp_ind'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open('../../pair_ind/sigcomp_ind/test_index.pkl', 'wb') as test_index_file:
        pickle.dump(pairs_paths,test_index_file)






def load_img(file_name1,file_name2,label,ext_h=820,ext_w=890):
    # CEDAR (829,890)
    # sigcomp (1100,2900)

    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=preprocess(img1,ext_h,ext_w)
    img2 = tf.io.read_file(file_name2, 'rb')  # 读取图片
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.rgb_to_grayscale(img2)
    img2=preprocess(img2,ext_h,ext_w)
    return img1,img2,label


def otsu(gray_img):
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    threshold_t = 0
    max_g = 0
    for t in range(255):
        n0 = gray_img[np.where(gray_img < t)]
        n1 = gray_img[np.where(gray_img>=t)]
        w0 = len(n0)/(h*w)
        w1 = len(n1)/(h*w)
        u0 = np.mean(n0) if len(n0)>0 else 0
        u1 = np.mean(n1) if len(n1)>0 else 0
        g = w0*w1*(u0-u1)**2
        if g > max_g :
            max_g = g
            threshold_t = t
    gray_img[gray_img>threshold_t] = 255
    return gray_img
if __name__ == '__main__':
    save_pairs()





