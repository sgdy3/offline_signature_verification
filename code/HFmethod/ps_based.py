import cv2
import numpy as np
from itertools import combinations
from scipy import ndimage
import esig
from sklearn.preprocessing import  minmax_scale
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

'''
================================
训练策略：
使用45位用户的真实签名作为训练集聚类获得码本；
10位用户作为测试用户；
测试集用户使用12张真实签名和熟练伪造签名作为模板；
================================
想多了，LNPS的实现十分简单，只需要将原始的Path除以length就可以了；
后续计算的时候自动就会将每一阶的PS进行“归一”并不需要单独求出每一阶PS再除以对应length的幂；
而由于下某一阶的LPS由同阶的PS进行线性组合得到，所以也和PS到LNPS的变换法相同。
具体参见pad上的笔记，与张量代数有关，目前没太看懂，只是通过归纳给出了变换规律。
================================
比较纠结的点在于window移动的步长是多少，是一个window的大小还是等于1呢？不足的部分是循环还是怎样？
假定是循环的吧！步长为window大小
================================
stride大小应该是1，也就是每个点都需要考虑周围一系列的点，作为主体考虑和作为客体考虑有区别，所以需要步长为1;
padding方面如果将签名视作是单纯的时间序列，不需要padding，因为时间序列本来就不同于传统图像，相邻点间不存在着相似性；
而做为轮廓而言，脱机签名由于缺乏起始点信息，无论从何处开始都是可以的，这就意味着padding可以认为是从该点开始的新轮廓。
感觉填充会好一点，考虑到步长为1，认为每个点都很重要。
'''


def joint_ps(img,degree):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=ndimage.gaussian_filter(gray_img,2)
    threshold,binarized_img=cv2.threshold(img,0,255,cv2.cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
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


org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'
np.random.seed(3)
train_user_order=np.random.choice(range(1,56),45,replace=False)
test_user_order=np.arange(1,56)[~np.isin(np.arange(1,56),train_user_order)] # 得到测试集用户

# training phase
sigs=[]
w=3 # pathlet-size
degree=3
for user in train_user_order:
    for id in range(1,25):
        img_path=org_path%(user,id)
        img=cv2.imread(img_path)
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=ndimage.gaussian_filter(gray_img,2)
        threshold,binarized_img=cv2.threshold(gray_img,0,255,cv2.cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
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
sigs=np.vstack(sigs)
rg=np.vstack([sigs.max(axis=0),sigs.min(axis=0)]) # 测试集归一化要用到
sigs=minmax_scale(sigs,axis=0)
M=24
model=KMeans(n_clusters=M,random_state=3)
model.fit(sigs)


'''
根据类内距离和类间距离的比值判断真实签名还是伪造签名
'''
# 用户相关判别，需要先得到用户模板的FM
# forg_template=[]
# org_template=[]
# template_num=10 # 真伪签名分别采纳10个作为模板
# template_sig_id=np.random.choice(range(1,24),template_num,replace=True)
# test_sig_id=np.arange(1,25)[~np.isin(np.arange(1,25),template_sig_id)]
#
# for user in test_user_order:
#     org_FM=np.zeros((template_num,M,M)) # 对一位用户存在着正负模板，需要分别记录FM,第一维为模板数目
#     forg_FM=np.zeros((template_num,M,M))
#     for ind,id in enumerate(template_sig_id):
#         org_img_path=org_path%(user,id)
#         org_img=cv2.imread(org_img_path)
#         org_sig=joint_ps(org_img, degree)
#         org_sig=np.array(org_sig).reshape(len(org_sig)*2,-1)
#         org_sig=(org_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
#         lab=model.predict(org_sig)
#         corcs=np.vstack([lab[::2],lab[1::2]]).T
#         for i in corcs:
#             org_FM[ind,i[0],i[1]]+=1
#         org_FM[ind,:,:]=org_FM[ind,:,:]/np.sum(org_FM[ind,:,:]) # normalize FM sum to 1
#
#         forg_img_path=forg_path%(user,id)
#         forg_img=cv2.imread(forg_img_path)
#         forg_sig=joint_ps(forg_img,degree)
#         forg_sig=np.array(forg_sig).reshape(len(forg_sig)*2,-1)
#         forg_sig=(forg_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
#         lab=model.predict(forg_sig)
#         corcs=np.vstack([lab[::2],lab[1::2]]).T
#         for i in corcs:
#             forg_FM[ind,i[0],i[1]]+=1
#         forg_FM[ind,:,:]=forg_FM[ind,:,:]/np.sum(forg_FM[ind,:,:])  # normalize FM sum to 1
#
#     org_template.append(np.expand_dims(org_FM,0))
#     forg_template.append(np.expand_dims(forg_FM,0))
# org_template=np.concatenate(org_template,axis=0)
# forg_template=np.concatenate(forg_template,axis=0)
#
# # 计算用户模板的类内距离
# org_inner_dist=[]
# forg_inner_dist=[]
# for user in range(org_template.shape[0]):
#     pairs=combinations(range(org_template.shape[1]),2)
#     dist=[]
#     for pair in pairs:
#         dist.append(np.linalg.norm(org_template[user,pair[0]]-org_template[user,pair[1]],ord=1)) # 统计所有类内组合间的距离
#     org_inner_dist.append(np.sum(dist))
# for user in range(forg_template.shape[0]):
#     pairs=combinations(range(forg_template.shape[1]),2)
#     dist=[]
#     for pair in pairs:
#         dist.append(np.linalg.norm(forg_template[user,pair[0]]-forg_template[user,pair[1]],ord=1)) # 统计所有类内组合间的距离
#     forg_inner_dist.append(np.sum(dist))
#
#
# pos_scores=[]
# neg_scores=[]
# for ind,user in enumerate(test_user_order):
#     org_FM=np.zeros((M,M)) # 对一位用户存在着正负模板，需要分别记录FM,第一维为模板数目
#     forg_FM=np.zeros((M,M))
#     for id in test_sig_id:
#         org_img_path=org_path%(user,id)
#         org_img=cv2.imread(org_img_path)
#         org_sig=joint_ps(org_img, degree)
#         org_sig=np.array(org_sig).reshape(len(org_sig)*2,-1)
#         org_sig=(org_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
#         lab=model.predict(org_sig)
#         corcs=np.vstack([lab[::2],lab[1::2]]).T
#         for i in corcs:
#             org_FM[i[0],i[1]]+=1
#         org_FM=org_FM/np.sum(org_FM) # normalize FM sum to 1
#
#         org_dist=[]
#         for j in range(org_template.shape[1]):
#             org_dist.append(np.linalg.norm(org_FM-org_template[ind,j],ord=1))
#         pos_scores.append((template_num-1)/2*np.sum(org_dist)/org_inner_dist[ind])
#
#         forg_dist=[]
#         for j in range(forg_template.shape[1]):
#             forg_dist.append(np.linalg.norm(org_FM-forg_template[ind,j],ord=1))
#         neg_scores.append((template_num-1)/2*np.sum(forg_dist)/forg_inner_dist[ind])
#
#
#         forg_img_path=forg_path%(user,id)
#         forg_img=cv2.imread(forg_img_path)
#         forg_sig=joint_ps(forg_img,degree)
#         forg_sig=np.array(forg_sig).reshape(len(forg_sig)*2,-1)
#         forg_sig=(forg_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
#         lab=model.predict(forg_sig)
#         corcs=np.vstack([lab[::2],lab[1::2]]).T
#         for i in corcs:
#             forg_FM[i[0],i[1]]+=1
#         forg_FM=forg_FM/np.sum(forg_FM)  # normalize FM sum to 1
#
#         org_dist=[]
#         for j in range(org_template.shape[1]):
#             org_dist.append(np.linalg.norm(forg_FM-org_template[ind,j],ord=1))
#         pos_scores.append((template_num-1)/2*np.sum(org_dist)/org_inner_dist[ind])
#
#         forg_dist=[]
#         for j in range(forg_template.shape[1]):
#             forg_dist.append(np.linalg.norm(forg_FM-forg_template[ind,j],ord=1))
#         neg_scores.append((template_num-1)/2*np.sum(forg_dist)/forg_inner_dist[ind])
# result=np.array(neg_scores)-np.array(pos_scores)
# result[np.where(result<0)]=0
# result[np.where(result>0)]=1
# labels=np.ones(result.shape)
# labels[1::2]=0
# print(f"accuracy:{(np.sum(labels==result)/labels.shape)[0]}")



'''
使用SVM训练用户相关分类器
'''
result=[]
# 用户相关判别，需要先得到用户模板的FM
forg_template=[]
org_template=[]
template_num=12  # 真实签名采纳10个作为模板
template_sig_id=np.random.choice(range(1,24),template_num,replace=False)


neg_vecs=[]
for user in train_user_order:
    for id in range(1,25):
        org_FM=np.zeros((M,M))
        org_img_path=org_path%(user,id)
        org_img=cv2.imread(org_img_path)
        org_sig=joint_ps(org_img, degree)
        org_sig=np.array(org_sig).reshape(len(org_sig)*2,-1)
        org_sig=(org_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
        lab=model.predict(org_sig)
        corcs=np.vstack([lab[::2],lab[1::2]]).T
        for i in corcs:
            org_FM[i[0],i[1]]+=1
        org_FM[:,:]=org_FM[:,:]/np.sum(org_FM[:,:]) # normalize FM sum to 1
        org_template.append(np.expand_dims(org_FM.flatten(),0))
neg_vecs=np.concatenate(org_template,axis=0)


forg_template=[]
org_template=[]
pred_lab=[]
result1=[]
for user in test_user_order:
    for id in range(1,25):
        org_FM=np.zeros((M,M)) # 对一位用户存在着正负模板，需要分别记录FM,第一维为模板数目
        forg_FM = np.zeros((M,M))
        org_img_path=org_path%(user,id)
        org_img=cv2.imread(org_img_path)
        org_sig=joint_ps(org_img, degree)
        org_sig=np.array(org_sig).reshape(len(org_sig)*2,-1)
        org_sig=(org_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
        lab=model.predict(org_sig)
        corcs=np.vstack([lab[::2],lab[1::2]]).T
        for i in corcs:
            org_FM[i[0],i[1]]+=1
        org_FM[:,:]=org_FM[:,:]/np.sum(org_FM[:,:]) # normalize FM sum to 1
        org_template.append(np.expand_dims(org_FM.flatten(),0))

        forg_img_path=forg_path%(user,id)
        forg_img=cv2.imread(forg_img_path)
        forg_sig=joint_ps(forg_img,degree)
        forg_sig=np.array(forg_sig).reshape(len(forg_sig)*2,-1)
        forg_sig=(forg_sig-rg[1])/(rg[0]-rg[1]) # 测试集归一化
        lab=model.predict(forg_sig)
        corcs=np.vstack([lab[::2],lab[1::2]]).T
        for i in corcs:
            forg_FM[i[0],i[1]]+=1
        forg_FM[:,:]=forg_FM[:,:]/np.sum(forg_FM[:,:])  # normalize FM sum to 1
        forg_template.append(np.expand_dims(forg_FM.flatten(),0))
org_template=np.concatenate(org_template,axis=0)
forg_template=np.concatenate(forg_template,axis=0)


for (ind,user) in enumerate(test_user_order):
    user_ind=np.arange(ind*24,(ind+1)*24)
    user_train_ind=np.random.choice(user_ind,template_num,replace=False)
    user_test_id=user_ind[~np.isin(user_ind,user_train_ind)]

    skew = neg_vecs.shape[0] / user_train_ind.shape[0]  # 不均匀样本平衡权重
    svm_input=np.vstack([neg_vecs,org_template[user_train_ind,:]])
    svm_label=np.concatenate([np.zeros(neg_vecs.shape[0]),np.ones(user_train_ind.shape[0])])
    svm=SVC(class_weight={1:skew},kernel='rbf')
    svm_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                         ('classifier', svm)])

    c_can = np.linspace(0.1, 1.1, 11)
    gamma_can = np.linspace(0.1, 1.1, 11)
    param_grid={'classifier__C': c_can}
    svr = GridSearchCV(svm_with_scaler,param_grid)
    svr.fit(svm_input,svm_label)
    hyper_dist=svr.best_estimator_.decision_function(np.concatenate([org_template[user_test_id,:],forg_template[user_ind,:]]))
    result.append(np.vstack([hyper_dist,np.concatenate([np.ones(user_test_id.shape[0]),np.zeros(user_ind.shape[0])])]))

    svm_init = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                         ('classifier', svm)])
    svm_init.fit(svm_input,svm_label)
    hyper_dist=svm_init.decision_function(np.concatenate([org_template[user_test_id,:],forg_template[user_ind,:]]))
    result1.append(np.vstack([hyper_dist,np.concatenate([np.ones(user_test_id.shape[0]),np.zeros(user_ind.shape[0])])]))


result=np.hstack(result).T
fpr, tpr, thresholds = roc_curve(result[:,1],result[:,0], pos_label=1)
fnr = 1 -tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
pred_label=result[:,0].copy()
pred_label[pred_label>eer_threshold]=1
pred_label[pred_label<=eer_threshold]=0
acc=(pred_label==result[:,1]).sum()/result.shape[0]

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



# img=cv2.imread(img_paths[0][0])
# gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img=ndimage.gaussian_filter(gray_img,2)
# threshold,binarized_img=cv2.threshold(gray_img,0,255,cv2.cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
# _,cnts,_ = cv2.findContours(binarized_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# img2=cv2.drawContours(img.copy(),cnts,-1,(126,234,243),1)
# new_cnts=[]
# img2=img.copy()
# area_least=4
# for c in cnts:
#     area = cv2.contourArea(c)
#     if(area<area_least):
#         print('pass')
#         continue
#     else:
#         points = cv2.approxPolyDP(c,1,True)
#         new_cnts.append(points)
#         img2=cv2.drawContours(img2,points,-1,(126,234,243),1)
# plt.imshow(img2,cmap='gray')
# img3=cv2.drawContours(img.copy(),new_cnts,-1,(126,234,243),1)
# sigs=[]
# for i in new_cnts:
#     temp=scale(np.squeeze(i),axis=0)
#     sig=esig.stream2sig(temp,5)
#     sigs.append(sig)