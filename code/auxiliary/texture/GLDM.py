# -*- coding: utf-8 -*-
# ---
# @File: GLDM.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/30
# Describe: 尝试编写代码获取图像的NGLDM
# ---


'''
2022/04/02
-------------------
GLDM的计算速度并不快，问题主要出现在im2col,占据了总运算时间的5/6，需要进行优化
已优化，快的飞起
'''

import time
import numpy as np



def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result

def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])

def _calculate_ij (rlmatrix):
    gray_level, run_length = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J+1
def _apply_over_degree(function, x1, x2):
    if function == np.divide:
        x2 = x2 + 1e-16
    result= function(x1[:, :], x2)
    result[result == np.inf] = 0
    result[np.isnan(result)] = 0
    return result

def GLDM(img,d=1,threshold=0,Ng=256,ignore_zero=True):
    img=np.pad(img,((d,d),(d,d)),constant_values=(-1,-1))  # 对原图进行padding，使得计算可以从边缘位置开始
    window_size=(2*d+1)
    #ravel=im2col(img,(window_size,window_size))
    ravel=im2col_sliding_broadcasting(img,(window_size,window_size))
    center=ravel[window_size+1,:]
    diff=ravel-center
    counts=(abs(diff)<=threshold).sum(axis=0) # 计算核范围内与中心差值绝对值小于阈值的点数目
    counts-=1  # 减去核中心的计数

    gldm=np.zeros((Ng,window_size**2))
    counts = np.vstack([center,counts])
    counts=counts.astype(int)
    freq=np.unique(counts,return_counts=True,axis=1)
    gldm[freq[0][0],freq[0][1]]=freq[1]
    if ignore_zero:
        gldm[0,:]=0
    return gldm

def gldm_features(img,d=1,threshold=0,Ng=256,ignore_zero=True):
    gldm=GLDM(img,d,threshold,Ng,ignore_zero)
    features = np.zeros(14,np.double)
    feature_label=["Small Dependence Emphasis","Large Dependence Emphasis","Gray Level Non-Uniformity",
                "Dependence Non-Uniformity","Dependence Non-Uniformity Normalized ","Gray Level Variance",
                "Dependence Variance","Dependence Entropy","Low Gray Level Emphasis","High Gray Level Emphasis",
                "Small Dependence Low Gray Level Emphasis","Small Dependence High Gray Level Emphasis",
                "Large Dependence Low Gray Level Emphasis","Large Dependence High Gray Level Emphasis"]

    I,J=_calculate_ij(gldm)
    # 如果没有忽略灰阶0的话，i就要从1开始，否则从哪0开始即可
    if not ignore_zero:
        I+=1
    S= np.sum(gldm)
    G = np.apply_over_axes(np.sum, gldm, axes=1)
    R = np.apply_over_axes(np.sum, gldm, axes=0)
    p=gldm/S
    epsilon=2.2*1e16

    # 感觉官网small dependence emphasis 的公式写错了，应该是除以J^2的
    features[0]=(np.apply_over_axes(np.sum,_apply_over_degree(np.divide,gldm,(J*J)),axes=(0,1))[0,0])/S
    features[1]=(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,gldm,(J*J)),axes=(0,1))[0,0])/S
    features[2]=(np.apply_over_axes(np.sum,(G*G), axes=(0,1))[0,0])/S
    features[3]=(np.apply_over_axes(np.sum,(R*R), axes=(0,1))[0,0])/S
    features[4]=(np.apply_over_axes(np.sum,(R*R), axes=(0,1))[0,0])/(S*S)

    I_mu=(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,p,I),axes=(0,1))[0,0])
    J_mu=(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,p,J),axes=(0,1))[0,0])
    features[5]=(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,p,(I-I_mu)*(I-I_mu)),axes=(0,1)))[0,0]
    features[6]=(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,p,(J-J_mu)*(J-J_mu)),axes=(0,1)))[0,0]
    features[7]=-(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,p,np.log(p+epsilon)),axes=(0,1)))[0,0]
    features[8]=(np.apply_over_axes(np.sum,_apply_over_degree(np.divide,gldm,(I*I)),axes=(0,1))[0,0])/S
    features[9]=(np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,gldm,(I*I)),axes=(0,1))[0,0])/S


    temp = _apply_over_degree(np.divide, gldm, (J*J))
    features[10]=(np.apply_over_axes(np.sum,_apply_over_degree(np.divide,temp,(I*I)),axes=(0,1))[0,0])/S
    features[11] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S)

    temp= _apply_over_degree(np.multiply, gldm, (J*J))
    features[12] = (np.apply_over_axes(np.sum,_apply_over_degree(np.divide,temp,(I*I)),axes=(0,1))[0,0])/S
    features[13] = (np.apply_over_axes(np.sum,_apply_over_degree(np.multiply,temp,(I*I)),axes=(0,1))[0,0])/S


    return  features,feature_label

if __name__=='__main__':
    test_arr=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])

