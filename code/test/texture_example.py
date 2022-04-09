# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/23
# Describe: 对pyfeat中的纹理特征函数进行验证和修改，以实现自己的需求
# ---


"""
更新日志：
=============
2022.03.23：
------------
1. 对pyfeat中的纹理特征相关函数进行了验证，具体见注释
2. 收集编写了自己的纹理特征矩阵计算代码
=============
"""


import numpy as np
from skimage.feature import greycomatrix, greycoprops
from itertools import groupby
import pyfeats as pf
from mahotas.features.texture import cooccurence
from pyfeats.textural.glrlm import glrlm_0
from pyfeats.textural.glszm import glszm
from pyfeats.textural.ngtdm import ngtdm
from skimage import measure

# def my_glcm(gray):
#     # get a gray level co-occurrence matrix (GLCM) using skimage
#     # parameters：the matrix of gray image，distance，direction，gray level，symmetric or not，standarzation or not
#     glcm = greycomatrix(gray, [1],[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
#                         256, symmetric = True, normed = True)
#
#     print(glcm.shape); print("===============================\n")
#
#     #获取共生矩阵的统计值.
#     feature=[]
#     for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
#         # 对比度，相异性，反方差矩阵，能量，相关系数，角二阶矩
#         temp = greycoprops(glcm, prop)
#         feature.append(temp)
#
#         print(prop, temp)
#         # print(prop + "_mean: ", np.mean(temp))
#         # print(prop + "_std:", np.std(temp, ddof = 1));
#         print( "==============================\n")
#     return feature


def my_glcm(input,d_x,d_y,Ng=256,sym=True):
    srcdata=input.copy()
    glcm=np.zeros((Ng,Ng))
    (height,width) = input.shape
    for j in range(height-d_y):
        for i in range(width-d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i+d_x]
            glcm[rows][cols]+=1  # 循环统计频次而已
            if sym:
                glcm[cols][rows]+=1
    return glcm


def my_glrlm(array, theta):
    '''
    计算给定图像的灰度游程矩阵
    参数：
    array: 输入，需要计算的图像
    theta: 输入，计算灰度游程矩阵时采用的角度，list类型，可包含字段:['deg0', 'deg45', 'deg90', 'deg135']
    glrlm: 输出，灰度游程矩阵的计算结果
    '''
    P = array
    x, y = P.shape
    min_pixels = np.min(P)   # 图像中最小的像素值
    run_length = max(x, y)   # 像素的最大游行长度
    num_level = np.max(P) - np.min(P) + 1   # 图像的灰度级数

    deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 原图分解按水平方向分解为多个序列
    deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 原图分解按竖直方向分解为多个序列
    diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   # 使用diag函数获取45°方向序列
    deg45 = [n.tolist() for n in diags]
    Pt = np.rot90(P, 3)   # 旋转后重复前一操作获取135°序列
    diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
    deg135 = [n.tolist() for n in diags]

    seqs={'deg0':deg0,'deg45':deg45,'deg90':deg90,'deg135':deg135}
    glrlm = np.zeros((num_level, run_length, len(theta)))
    for angle in theta:
        for splitvec in range(0, len(seqs)):
            flattened =seqs[angle][splitvec] # 获取某一方向上的某个序列
            answer = []
            for key, iter in groupby(flattened):   # 使用itertools.groupby()获取延续灰度值长度
                answer.append((key, len(list(iter))))
            for ansIndex in range(0, len(answer)):
                glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1   # 每次将统计像素值减去最小值就可以填入GLRLM矩阵中
    return glrlm

def my_glszm(f,mask,Ng=256,ignore_zero=True,connectivity=2):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    Ng: int,default = 256
        maximum image level
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.

    Returns
    -------
    GLSZM : numpy ndarray
        GLSZ Matrix.
    '''
    levels=np.arange(0,Ng)

    Ns = f.shape[0] * f.shape[1]
    if ignore_zero:
        GLSZM = np.zeros((Ng-1,Ns), np.double)
    else:
        GLSZM = np.zeros((Ng,Ns), np.double)
    start_level=1 if ignore_zero else 0

    temp = f.copy()
    for i in range(start_level,Ng):
        temp[f!=levels[i]] = 0
        temp[f==levels[i]] = 1
        connected_components = measure.label(temp, connectivity=connectivity)
        connected_components = connected_components * mask
        nZone = len(np.unique(connected_components))
        for j in range(1,nZone): # ignore the "0" element in connected_components
            col = np.count_nonzero(connected_components==j)
            # the fist column of the GLSZM represent the num of connected domain with area1
            # the fist row of the GLSZM represent gray level 0 or 1 depending on "ignore_zero"
            GLSZM[i-start_level,col-1] += 1

    return GLSZM

'''
1. 通过最外层API计算了基于图像的GLCM的14个纹理特征
2. 通过内层API计算了0方向的GLCM
最外层API给出的特征是0,45,90,135四个方向GLRLM矩阵提取特征的均值，以及max-min
此外默认是对称的，也就是0°=180°，21和12等价。
以下验证了矩阵计算结果正确，无需自己编辑函数
'''
test_arr=np.array([[1,2,5,2,3],[3,2,1,3,1],[1,3,5,5,2],[1,1,1,1,2],[1,2,4,3,5]])
glrm_mean, glrm_range, glrm_mean_lab, glrm_range_lab=pf.glcm_features(test_arr,True)  # feature_range为统计量max-min的值
glrmat=cooccurence(test_arr,0)

'''
1. 通过最外层API计算了基于图像的GLRLM游程的11个纹理特征
2. 通过内层API计算了0方向的GLRLM
值得注意最外层API给出的特征是0,45,90,135四个方向GLRLM矩阵提取特征的均值；
此外默认灰度级为0~255的256级，计算各方向GLRLM时忽略灰阶为0那一行，再此外，mask参数是无效的
以下验证了矩阵计算结果正确，无需自己编辑函数
'''
test_arr=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])
glrlm_feature,glrlm_lab=pf.glrlm_features(test_arr,None)
glrlm_f0=glrlm_0(test_arr,np.ones(test_arr.shape),5,max(test_arr.shape),False)

'''
1. 通过最外层API计算了基于图像的GLSZM的14个纹理特征
2. 通过内层API计算了GLSZM
很不凑巧，这个函数有问题：
1. 连通域默认为4连通域
2. 默认gray level 256，但在计算时255没有计算进去
3. 返回的矩阵第一行是任何时刻都为零的;
4. 统计了某个灰阶所有连通域时，还将图中非该灰阶的部分统计了并作为对应面积大小的，出现频率为1的连通域存在
修改后的mat计算方式放在上方my_glszm()
'''
test_arr=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])
test_arr=test_arr+(255-4)
glszm_feature,glszm_lab=pf.glszm_features(test_arr,None)
glszmat=glszm(test_arr,np.ones(test_arr.shape))

'''
1. 通过最外层API计算了基于图像的NGTD的5个纹理特征
2. 通过内层API计算了NGTD
很不凑巧，这个函数有问题：
1. NGTD的计算与卷积类似，同样存在着边缘的问题，这里的解决方案是直接去除无法进行完全计算的区域
   即只对原图（h-2*d,y-2*d）区域统计（假定差分范围为2*d+1的矩形框）
2. mask的处理是：所有差分区域包含masked pixel的都不纳入统计，一般一个masked点引起共(2*d+1)^2个不纳入统计
'''
test_arr=np.array([[1,2,5,2],[3,5,1,3],[1,3,5,5],[3,1,1,1]])
ngtd_feature,ngtd_labels=pf.ngtdm_features(test_arr,np.ones(test_arr.shape))
temp=np.ones(test_arr.shape)
temp[1,2]=0
ngtdmat=ngtdm(test_arr,temp,1,6)