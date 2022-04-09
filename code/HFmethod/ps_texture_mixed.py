# -*- coding: utf-8 -*-
# ---
# @File: ps_texture_mixed.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/4/2
# Describe:联合PS和texture特征做出判决
# ---

'''
20222/04/02
-------------------
提取PS特征和texture mat特征在此前已经研究过了，后者比较好处理，因为一张图片能够产生的纹理特征数目是固定的
但由于采用的是脱机图像，PS特征是对图像的多边形轮廓提取的，每个轮廓能够提取的PS特征是固定的，但可提取出的
轮廓数目不同，这一点有点难以解决，之前是通过BOW方法提取的，这里也先试试吧，那就需要训练集来获取聚类中心了

20222/04/03
-------------------
all right,代码完成了，使用45个用户作为训练集来获取码书，同时题目真实签名的BOW特征和texture特征作为负样本；
测试集10个用户中每个都选取14个签名作为正样本，分别训练SVM，其余34个签名用来测试。效果还不错吧，就是感觉
纹理特征被耽搁了，本来是不用45个用户作为训练的，但要提取码书就没办法了。
目前问题：
1. 能不能不提取码书，对长度不一的PS特征提取某些表征使得输出特征是固定长度的？
'''

import cv2
import numpy as np
import esig
from auxiliary.texture_mat import ngtdm_features
from auxiliary.texture.GLDM import gldm_features
import pyfeats as pf
from auxiliary.moment_preprocess import denoise
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
from sklearn.svm import SVC
import sklearn.pipeline as pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def ps_feature(img,w,degree):
    sigs=[]
    binarized_img=img.copy()
    binarized_img[binarized_img>0]=255
    _,cnts,_ = cv2.findContours(binarized_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_cnts=[]
    min_area=4
    for c in cnts:
        area = cv2.contourArea(c) # 面积较小的轮廓抛弃
        if(area<min_area):
            continue
        else:
            points = cv2.approxPolyDP(c,1,True)
            if(points.shape[0]<w): # 小于pathlet size的抛弃
                continue
            else:
                new_cnts.append(points)
    for cnt in new_cnts:
        cnt=np.squeeze(cnt).astype(float)
        # 好像有的算法是不进行padding
        padding=False # 填充
        if padding:
            padding_points=w-cnt.shape[0]%w
            cnt=np.vstack([cnt,cnt[:padding_points]])  # path末尾不足，循环补齐
        for k in range(cnt.shape[0]-w+1): # window大小4，步长1，类似卷积计算公式
            path=cnt[k:k+w]
            length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum() # 计算path的长度
            sig=esig.stream2logsig(path/length,degree)  # 尺度归一化
            sigs.append(sig)
    return sigs


def joint_ps(img,degree):
    binarized_img=img.copy()
    binarized_img[binarized_img>0]=255
    _,cnts,_ = cv2.findContours(binarized_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_cnts=[]
    min_area=4
    sigs=[]
    for c in cnts:
        area = cv2.contourArea(c) # 面积较小的轮廓抛弃
        if(area<min_area):
            continue
        else:
            points = cv2.approxPolyDP(c,1,True)
            if(points.shape[0]<2*w-1): # 小于pathlet size的抛弃
                continue
            else:
                new_cnts.append(points)
    for cnt in new_cnts:
        cnt=np.squeeze(cnt).astype(float)
        padding=False # 填充
        joint_w=2*w-1
        if padding:
            padding_points=joint_w-cnt.shape[0]%joint_w
            cnt=np.vstack([cnt,cnt[:padding_points]])  # path末尾不足，循环补齐
        for k in range(cnt.shape[0]-joint_w+1): # window大小4，步长1，类似卷积计算公式
            joint_path=cnt[k:k+joint_w]
            path1=joint_path[:w]
            path2=joint_path[w-1:]
            length1=np.linalg.norm(np.diff(path1,axis=0),axis=1).sum()
            sig1=esig.stream2logsig(path1/length1,degree)  # 尺度归一化
            length2=np.linalg.norm(np.diff(path2,axis=0),axis=1).sum()
            sig2=esig.stream2logsig(path2/length2,degree)  # 尺度归一化
            sigs.append([sig1,sig2])
    return sigs



def texture_feature(img):
    glcm_f,_,_,_=pf.glcm_features(img,ignore_zeros=True)
    glrlm_f,_=pf.glrlm_features(img,np.ones(img.shape))
    ngtd_f,_=ngtdm_features(img,np.ones(img.shape))
    gldm_f,_=gldm_features(img)
    texture_vec=np.concatenate([glcm_f,glrlm_f,ngtd_f,gldm_f])
    return texture_vec


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
             lw=lw, label='ROC curve (area = %0.6f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()



org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

user_num=55
sig_num=24
M=12
template_num=14
load=True
np.random.seed(3)
train_user_order=np.random.choice(range(1,56),45,replace=False)
test_user_order=np.arange(1,56)[~np.isin(np.arange(1,56),train_user_order)] # 得到测试集用户


# 读取或计算所有用户的纹理特征
if not load:
    org_texture_vecs=[]
    forg_texture_vecs=[]
    for user in range(1,user_num+1):
        start=time.perf_counter()
        for id in range(1,sig_num+1):
            img=cv2.imread(org_path%(user,id),0)
            img=denoise(img,reigon_mask=True)
            img=255-img
            org_texture_f=texture_feature(img)
            org_texture_vecs.append(org_texture_f)

            img=cv2.imread(forg_path%(user,id),0)
            img=denoise(img,reigon_mask=True)
            img=255-img
            forg_texture_f=texture_feature(img)
            forg_texture_vecs.append(forg_texture_f)
        print(f"{user} donwn")
        end=time.perf_counter()
        print(f"用时：{end-start}")

    org_texture_vecs=np.array(org_texture_vecs)
    forg_texture_vecs=np.array(forg_texture_vecs)
    np.save('../../data/texture_mat/ori_org.npy',org_texture_vecs)
    np.save('../../data/texture_mat/ori_forg.npy',forg_texture_vecs)
else:
    org_texture_vecs=np.load('../../data/texture_mat/ori_org.npy')
    forg_texture_vecs=np.load('../../data/texture_mat/ori_forg.npy')
print("用户纹理特征计算完成")

# 不同图片的PS特征维数不一，且计算量不大，就不保存了。
org_ps_vecs=[]
forg_ps_vecs=[]
train_ps_features=[]
w=3
degree=3
for user in train_user_order:
    for id in range(1,sig_num+1):
        img=cv2.imread(org_path%(user,id),0)
        img=denoise(img,reigon_mask=True)
        img=255-img
        ps_f=ps_feature(img,w,degree)
        train_ps_features= train_ps_features + ps_f
train_ps_features=np.vstack(train_ps_features)

ps_scaler=MinMaxScaler()
ps_scaler.fit(train_ps_features)
train_ps_features=ps_scaler.transform(train_ps_features)
cluster_model=KMeans(n_clusters=M,random_state=3)
cluster_model.fit(train_ps_features)
print("基于训练集PS特征构建码书完成")
del train_ps_features


# 利用训练集构建负样本（计算训练集用户真实的PS特征）
neg_bow_features=[]
neg_texture_features=[]
for user in train_user_order:
    for id in range(1,sig_num+1):
        # ps特征矩阵
        org_FM=np.zeros((M,M))
        org_img_path=org_path%(user,id)
        org_img=cv2.imread(org_img_path,0)
        org_img=denoise(org_img)
        org_img=255-org_img
        org_sig=joint_ps(org_img, degree)
        org_sig=np.array(org_sig).reshape(len(org_sig)*2,-1)

        org_sig=ps_scaler.transform(org_sig) # 训练集归一化
        lab=cluster_model.predict(org_sig)
        corcs=np.vstack([lab[::2],lab[1::2]]).T
        for i in corcs:
            org_FM[i[0],i[1]]+=1
        org_FM[:,:]=org_FM[:,:]/np.sum(org_FM[:,:]) # normalize FM sum to 1
        neg_ps_FM=org_FM.flatten()
        neg_bow_features.append(np.expand_dims(neg_ps_FM, 0))

        # 纹理特征
        neg_texture=org_texture_vecs[(user-1)*sig_num+id-1,:]
        neg_texture_features.append(neg_texture)
neg_bow_features=np.concatenate(neg_bow_features)
neg_texture_features=np.vstack(neg_texture_features)
print('训练集上负样本特征构建完成')


# 负样本（训练集）上纹理特征归一化、降维
texture_processor=pipeline.Pipeline([('scaler',MinMaxScaler()),
                                     ('reduction',PCA(n_components=15))])
texture_processor.fit(neg_texture_features)
neg_texture_features=texture_processor.transform(neg_texture_features)

# 负样本（训练集）上bow特征降维
reduction_ps=PCA(n_components=20)
reduction_ps.fit(neg_bow_features)
neg_bow_features=reduction_ps.transform(neg_bow_features)

neg_vecs=np.hstack([neg_bow_features, neg_texture_features])
print('训练集上负样本特征处理完成')

# 计算测试集用户的PS特征
test_org_bow=[]
test_forg_bow=[]
test_org_texture=[]
test_forg_texture=[]
pred_lab=[]
result1=[]
for user in test_user_order:
    for id in range(1,sig_num+1):
        # 测试集用户真实和虚假签名的PS特征
        org_FM=np.zeros((M,M))  # 对一位用户存在着正负模板，需要分别记录FM,第一维为模板数目
        forg_FM = np.zeros((M,M))
        org_img_path=org_path%(user,id)
        org_img=cv2.imread(org_img_path,0)
        org_img=denoise(org_img)
        org_img=255-org_img
        org_sig=joint_ps(org_img, degree)
        org_sig=np.array(org_sig).reshape(len(org_sig)*2,-1)

        org_sig=ps_scaler.transform(org_sig)  # 测试集归一化
        lab=cluster_model.predict(org_sig)
        corcs=np.vstack([lab[::2],lab[1::2]]).T
        for i in corcs:
            org_FM[i[0],i[1]]+=1
        org_FM[:,:]=org_FM[:,:]/np.sum(org_FM[:,:]) # normalize FM sum to 1
        test_org_bow.append(np.expand_dims(org_FM.flatten(), 0))

        forg_img_path=forg_path%(user,id)
        forg_img=cv2.imread(forg_img_path,0)
        forg_img=denoise(forg_img)
        forg_img=255-forg_img
        forg_sig=joint_ps(forg_img,degree)
        forg_sig=np.array(forg_sig).reshape(len(forg_sig)*2,-1)
        forg_sig=ps_scaler.transform(forg_sig) # 测试集归一化
        lab=cluster_model.predict(forg_sig)
        corcs=np.vstack([lab[::2],lab[1::2]]).T
        for i in corcs:
            forg_FM[i[0],i[1]]+=1
        forg_FM[:,:]=forg_FM[:,:]/np.sum(forg_FM[:,:])  # normalize FM sum to 1
        test_forg_bow.append(np.expand_dims(forg_FM.flatten(), 0))

        # 测试集用户真实和虚假签名的纹理特征
        test_org_texture.append(org_texture_vecs[(user - 1) * sig_num + id, :])
        test_forg_texture.append(forg_texture_vecs[(user - 1) * sig_num + id, :])
print("测试集上特征提取完成")


# 测试集上bow特征降维
test_org_bow=np.concatenate(test_org_bow, axis=0)
test_org_bow=reduction_ps.transform(test_org_bow)
test_forg_bow=np.concatenate(test_forg_bow, axis=0)
test_forg_bow=reduction_ps.transform(test_forg_bow)

# 测试集上纹理特征归一化、降维
test_org_texture=np.vstack(test_org_texture)
test_org_texture=texture_processor.transform(test_org_texture)
test_forg_texture=np.vstack(test_forg_texture)
test_forg_texture=texture_processor.transform(test_forg_texture)

org_template=np.hstack([test_org_bow, test_org_texture])
forg_template=np.hstack([test_forg_bow, test_forg_texture])
print("测试集上特征处理完成")


# 用户相关判决阶段
cotrast=[]
score=[]
result=[]
label=[]
for (ind,user) in enumerate(test_user_order):
    user_ind=np.arange(ind*sig_num,(ind+1)*sig_num)
    pos_train=np.random.choice(user_ind,template_num,replace=False)
    neg_train=np.random.choice(range(neg_vecs.shape[0]),template_num,replace=False)

    test_vecs=np.concatenate([org_template[user_ind[~np.isin(user_ind,pos_train)]],forg_template[user_ind]])
    train_vecs=np.concatenate([org_template[pos_train],neg_vecs[neg_train]])
    train_label=np.concatenate([np.ones(template_num),np.zeros(template_num)])
    test_label=np.concatenate([np.ones(sig_num-template_num),np.zeros(sig_num)])


    verif=SVC(C=1.2,kernel='rbf')
    verif.fit(train_vecs,train_label)

    hyper_dist=verif.decision_function(test_vecs)
    score.append(verif.score(test_vecs,test_label))
    print(f"user {user+1}: scores {score[-1]}")
    result.append(hyper_dist)
    label.append(test_label)
cotrast.append(np.mean(score))
label=np.concatenate(label)
result=np.concatenate(result)
curve_eval(label,result)








