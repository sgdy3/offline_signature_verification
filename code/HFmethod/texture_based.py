# -*- coding: utf-8 -*-
# ---
# @File: texture_based.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/23
# Describe: offline signature verification system based on textual feature.
# ---


'''
2022/3/23
-------------------
初步设定目标为先对每个签名都提取纹理特征，获得纹理特征矩阵后再对每个用户建模判断真伪
现存问题：
1. 计算耗时过大，不可能一次提取出所有签名的纹理特征

2022/03/27
------------------
发现glcm的平均计算耗时远低于其他特征，ngtd其次，glrlm再次之，glszm的平均耗时远高于前三者
一个用户的签名要跑3.5min,glcm用时0.68s,ngtd用时11.74s，glrlm用时20.43s，glszm用时2.8min

2022/03/29
------------------
在StackOverflow上查到的结果是特征选择并不能优化SVM的效果，甚至是有害的。那现在面临的问题是
1. 是否要剔除glszm特征，
2. 是否要对特征降维
确定了两件事：
1. 纹理特征之间确实是高度相关的，通过计算特征彼此间的相关系数可以证明这一点，但glcm,glrlm,ngtd彼此间的相关性
   没有那么强，这可能意味着我们确实又需要求取所有特征而不能舍弃。
2. resize与否对于特征的区别性有着很大的影响

2022/04/02
------------------
目前结论：
1. 特征一起用比单独用或者组合任意俩个要好，但glszm还是太费时，舍弃吧。
2. PCA降维很有用
3. 不要resize后提取纹理特征

'''

import cv2
import numpy as np
from sklearn.svm import SVC
from auxiliary.moment_preprocess import moment_preprocess
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pyfeats as pf
from auxiliary.moment_preprocess import denoise
from auxiliary.texture_mat import glszm_features
from auxiliary.texture_mat import ngtdm_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time
from auxiliary.GLDM import gldm_features
from itertools import combinations


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

def feature_extraction(user,id):
    org_ind=org_path%(user,id)
    org_sig=cv2.imread(org_ind,0)
    # org_sig=moment_preprocess(org_sig)
    org_sig=denoise(org_sig)
    org_sig=255-org_sig

    start=time.perf_counter()
    glcm_f,_,_,_=pf.glcm_features(org_sig,ignore_zeros=True)
    end=time.perf_counter()
    glcm_time.append((end-start))
    print(f"glcm用时：{end-start}s")

    start=time.perf_counter()
    glrlm_f,_=pf.glrlm_features(org_sig,np.ones(org_sig.shape))
    end=time.perf_counter()
    glrlm_time.append((end-start))
    print(f"glrlm用时：{end-start}s")

    # start=time.perf_counter()
    # glszm_f,_=glszm_features(org_sig,np.ones(org_sig.shape))
    # end=time.perf_counter()
    # glszm_time.append((end-start))
    # print(f"glszm用时：{end-start}s")

    start=time.perf_counter()
    ngtd_f,_=ngtdm_features(org_sig,np.ones(org_sig.shape))
    end=time.perf_counter()
    ngtd_time.append((end-start))
    print(f"ngtdm用时：{end-start}s")

    start=time.perf_counter()
    gldm_f=gldm_features(org_sig)
    end=time.perf_counter()
    ngtd_time.append((end-start))
    print(f"gldm用时：{end-start}s")

    #org_vec=np.concatenate([glcm_f,glrlm_f,glszm_f,ngtd_f])
    org_vec=np.concatenate([glcm_f,glrlm_f,ngtd_f])

    forg_ind=forg_path%(user,id)
    forg_sig=cv2.imread(forg_ind,0)
    # forg_sig=moment_preprocess(forg_sig)
    forg_sig=denoise(forg_sig)
    forg_sig=255-forg_sig
    glcm_f,_,_,_=pf.glcm_features(forg_sig,ignore_zeros=True)
    glrlm_f,_=pf.glrlm_features(forg_sig,np.ones(forg_sig.shape))
    # glszm_f,_=glszm_features(forg_sig,np.ones(forg_sig.shape))
    ngtd_f,_=ngtdm_features(forg_sig,np.ones(forg_sig.shape))
    # forg_vec=np.concatenate([glcm_f,glrlm_f,glszm_f,ngtd_f])
    forg_vec=np.concatenate([glcm_f,glrlm_f,ngtd_f])

    return org_vec,forg_vec


org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
user_num=55
sig_num=24
template_num=14
org_inds=[]
forg_inds=[]


'''
首先对每个用户的每个样本都提取出纹理特征，再统一进行处理
'''

glcm_time=[]
glrlm_time=[]
glszm_time=[]
ngtd_time=[]
gldm_time=[]

# user=7
# id=6
# org_vec,forg_vec=feature_extraction(user,id)

load=True
if not load:
    org_vecs=[]
    forg_vecs=[]
    for user in range(1,user_num+1):
        start=time.perf_counter()
        for id in range(1,sig_num+1):
            org_vec,forg_vec=feature_extraction(user,id)
            org_vecs.append(org_vec)
            forg_vecs.append(forg_vec)
        print(f"{user} donwn")
        end=time.perf_counter()
        print(f"用时：{end-start}")

    glszm_time=np.array(glszm_time)
    glrlm_time=np.array(glrlm_time)
    ngtd_time=np.array(ngtd_time)
    glcm_time=np.array(glcm_time)
    org_vecs=np.array(org_vecs)
    forg_vecs=np.array(forg_vecs)
else:
    org_vecs=np.load('../../data/texture_mat/ori_org.npy')
    forg_vecs=np.load('../../data/texture_mat/ori_forg.npy')

save_org=org_vecs.copy()
save_forg=forg_vecs.copy()

cotranst=[]

score=[]
result=[]
for user in range(user_num):
    '''
    对于每个用户，随机挑选template_num个真实签名作为正样本
    template_num个其他用户的真实签名作为负样本，
    '''
    user_ind=np.arange(user*sig_num,(user+1)*sig_num)
    other_ind=np.arange(org_vecs.shape[0])[~np.isin(np.arange(org_vecs.shape[0]),user_ind)]
    pos_train=np.random.choice(user_ind,template_num,replace=False)
    neg_train=np.random.choice(other_ind,template_num,replace=False)

    test_vecs=np.concatenate([org_vecs[user_ind[~np.isin(user_ind,pos_train)]],forg_vecs[user_ind]])
    train_vecs=np.concatenate([org_vecs[pos_train],org_vecs[neg_train]])
    train_label=np.concatenate([np.ones(template_num),np.zeros(template_num)])
    test_label=np.concatenate([np.ones(sig_num-template_num),np.zeros(sig_num)])

    norm=MinMaxScaler()
    norm.fit(train_vecs)
    train_vecs=norm.transform(train_vecs)
    test_vecs=norm.transform(test_vecs)

    # 尚不确定是否有需要对不同纹理特征降维
    # LowDim=PCA(n_components=10)
    # LowDim.fit(train_vecs)
    # train_vecs=LowDim.transform(train_vecs)
    # test_vecs=LowDim.transform(test_vecs)


    verif=SVC(C=1.2,kernel='rbf')
    verif.fit(train_vecs,train_label)

    hyper_dis=verif.decision_function(test_vecs)
    score.append(verif.score(test_vecs,test_label))
    # print(f"user {user+1}: scores {score[-1]}")
    result.append(hyper_dis)
cotranst.append(np.mean(score))

# feature={'glcm':range(0,14),'glrlm':range(14,25),'ngtdm':range(25,30)}
# for i in feature.keys():
#     print(i)
#     org_vecs=save_org[:,feature.get(i)].copy()
#     forg_vecs=save_forg[:,feature.get(i)].copy()
#
#     score=[]
#     result=[]
#     for user in range(user_num):
#         '''
#         对于每个用户，随机挑选template_num个真实签名作为正样本
#         template_num个其他用户的真实签名作为负样本，
#         '''
#         user_ind=np.arange(user*sig_num,(user+1)*sig_num)
#         other_ind=np.arange(org_vecs.shape[0])[~np.isin(np.arange(org_vecs.shape[0]),user_ind)]
#         pos_train=np.random.choice(user_ind,template_num,replace=False)
#         neg_train=np.random.choice(other_ind,template_num,replace=False)
#
#         test_vecs=np.concatenate([org_vecs[user_ind[~np.isin(user_ind,pos_train)]],forg_vecs[user_ind]])
#         train_vecs=np.concatenate([org_vecs[pos_train],org_vecs[neg_train]])
#         train_label=np.concatenate([np.ones(template_num),np.zeros(template_num)])
#         test_label=np.concatenate([np.ones(sig_num-template_num),np.zeros(sig_num)])
#
#         norm=MinMaxScaler()
#         norm.fit(train_vecs)
#         train_vecs=norm.transform(train_vecs)
#         test_vecs=norm.transform(test_vecs)
#
#         # 尚不确定是否有需要对不同纹理特征降维
#         # LowDim=PCA(n_components=10)
#         # LowDim.fit(train_vecs)
#         # train_vecs=LowDim.transform(train_vecs)
#         # test_vecs=LowDim.transform(test_vecs)
#
#
#         verif=SVC(C=1.2,kernel='rbf')
#         verif.fit(train_vecs,train_label)
#
#         hyper_dis=verif.decision_function(test_vecs)
#         score.append(verif.score(test_vecs,test_label))
#         # print(f"user {user+1}: scores {score[-1]}")
#         result.append(hyper_dis)
#     cotranst.append(np.mean(score))
#
#
#
# for i in combinations(feature.keys(),2):
#     print(i)
#     org_vecs=np.hstack([save_org[:,feature[i[0]]],save_org[:,feature[i[1]]]])
#     forg_vecs=np.hstack([save_forg[:,feature[i[0]]],save_forg[:,feature[i[1]]]])
#
#     score=[]
#     result=[]
#     for user in range(user_num):
#         '''
#         对于每个用户，随机挑选template_num个真实签名作为正样本
#         template_num个其他用户的真实签名作为负样本，
#         '''
#         user_ind=np.arange(user*sig_num,(user+1)*sig_num)
#         other_ind=np.arange(org_vecs.shape[0])[~np.isin(np.arange(org_vecs.shape[0]),user_ind)]
#         pos_train=np.random.choice(user_ind,template_num,replace=False)
#         neg_train=np.random.choice(other_ind,template_num,replace=False)
#
#         test_vecs=np.concatenate([org_vecs[user_ind[~np.isin(user_ind,pos_train)]],forg_vecs[user_ind]])
#         train_vecs=np.concatenate([org_vecs[pos_train],org_vecs[neg_train]])
#         train_label=np.concatenate([np.ones(template_num),np.zeros(template_num)])
#         test_label=np.concatenate([np.ones(sig_num-template_num),np.zeros(sig_num)])
#
#         norm=MinMaxScaler()
#         norm.fit(train_vecs)
#         train_vecs=norm.transform(train_vecs)
#         test_vecs=norm.transform(test_vecs)
#
#         # 尚不确定是否有需要对不同纹理特征降维
#         # LowDim=PCA(n_components=10)
#         # LowDim.fit(train_vecs)
#         # train_vecs=LowDim.transform(train_vecs)
#         # test_vecs=LowDim.transform(test_vecs)
#
#
#         verif=SVC(C=1.2,kernel='rbf')
#         verif.fit(train_vecs,train_label)
#
#         hyper_dis=verif.decision_function(test_vecs)
#         score.append(verif.score(test_vecs,test_label))
#         # print(f"user {user+1}: scores {score[-1]}")
#         result.append(hyper_dis)
#     cotranst.append(np.mean(score))
#
# print(cotranst)