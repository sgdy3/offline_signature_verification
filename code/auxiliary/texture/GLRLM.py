# -*- coding: utf-8 -*-
# ---
# @File: GLRLM.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/4/9
# Describe: 算了，还是自己实现一遍GLCM吧，pyfeats里的自定义化的程度太低了
# ---


'''
2022/04/10
结果表面自己的方法在Ng=256时的运算速率略高于原始的方法（25s:24s）；
当Ng较小时，运算速率大幅提升，且Ng越小提升越显著(2.5s:15s)。
'''

import numpy as np
import itertools


def my_glrlm_0(f,Ng=5,Nr=5,ignore_zero=True):
    """
     Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    Ng: int,default = 256
        maximum image level
    Nr：run length
    Returns
    -------
    degreeMatrix: numpy ndarray
        GLRLM0
    当灰阶数较少时，效率很高
    """
    degreeMatrix=np.zeros((Ng+1,Nr))
    f=np.hstack([f,np.ones((f.shape[0],1))*Ng]).astype(int)
    deg0=f.flatten()
    count=[]
    for k,v in itertools.groupby(deg0):
        count.append([k,len(list(v))])
    count=np.array(count).T
    count[1,:]-=1
    ind,times=np.unique(count,return_counts=True,axis=1)
    degreeMatrix[ind[0],ind[1]]=times
    degreeMatrix=degreeMatrix[:Ng,:]
    return degreeMatrix[1:,:] if ignore_zero else degreeMatrix




def my_glrlm_90(f,Ng=5,Nr=5,ignore_zero=True):
    """
     Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    Ng: int,default = 256
        maximum image level
    Nr：run length
    Returns
    -------
    degreeMatrix: numpy ndarray
        GLRLM0
    当灰阶数较少时，效率很高
    """
    f=f.T
    degreeMatrix=np.zeros((Ng+1,Nr))
    f=np.hstack([f,np.ones((f.shape[0],1))*Ng]).astype(int)
    deg90=f.flatten()
    count=[]
    for k,v in itertools.groupby(deg90):
        count.append([k,len(list(v))])
    count=np.array(count).T
    count[1,:]-=1
    ind,times=np.unique(count,return_counts=True,axis=1)
    degreeMatrix[ind[0],ind[1]]=times
    degreeMatrix=degreeMatrix[:Ng,:]
    return degreeMatrix[1:,:] if ignore_zero else degreeMatrix


def my_glrlm_45(f,Ng=5,Nr=5,ignore_zero=True):
    """
     Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    Ng: int,default = 256
        maximum image level
    Nr：run length
    Returns
    -------
    degreeMatrix: numpy ndarray
        GLRLM0
    当灰阶数较少时，效率很高
    """
    diags = [f[::-1, :].diagonal(i) for i in range(-f.shape[0]+1, f.shape[1])]   # 使用diag函数获取45°方向序列
    deg45= [n.tolist()+[Ng] for n in diags]
    degreeMatrix=np.zeros((Ng+1,Nr))
    deg45=np.hstack(deg45)
    count=[]
    for k,v in itertools.groupby(deg45):
        count.append([k,len(list(v))])
    count=np.array(count).T
    count[1,:]-=1
    ind,times=np.unique(count,return_counts=True,axis=1)
    degreeMatrix[ind[0],ind[1]]=times
    degreeMatrix=degreeMatrix[:Ng,:]
    return degreeMatrix[1:,:] if ignore_zero else degreeMatrix




def my_glrlm_135(f,Ng=5,Nr=5,ignore_zero=True):
    """
     Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    Ng: int,default = 256
        maximum image level
    Nr：run length
    Returns
    -------
    degreeMatrix: numpy ndarray
        GLRLM0
    当灰阶数较少时，效率很高
    """
    ft = np.rot90(f, 3)   # 旋转后重复前一操作获取135°序列
    diags = [ft[::-1, :].diagonal(i) for i in range(-ft.shape[0]+1, ft.shape[1])]
    deg135 = [n.tolist()+[Ng] for n in diags]
    degreeMatrix=np.zeros((Ng+1,Nr))
    deg135=np.hstack(deg135)
    count=[]
    for k,v in itertools.groupby(deg135):
        count.append([k,len(list(v))])
    count=np.array(count).T
    count[1,:]-=1
    ind,times=np.unique(count,return_counts=True,axis=1)
    degreeMatrix[ind[0],ind[1]]=times
    degreeMatrix=degreeMatrix[:Ng,:]
    return degreeMatrix[1:,:] if ignore_zero else degreeMatrix

def _apply_over_degree(function, x1, x2):
    if function == np.divide:
        x2 = x2 + 1e-16
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        result[:, :, i] = function(x1[:, :, i], x2)
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
    return result

def _calculate_ij (rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J+1

def _calculate_s(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

def my_glrlm(f,Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.

    Returns
    -------
    mat : numpy ndarray
        GLRL Matrices for 0, 45, 90 and 135 degrees.
    '''
    runLength = max(f.shape)
    mat0 = my_glrlm_0(f, Ng=Ng, Nr=runLength,ignore_zero=True)
    mat45 = my_glrlm_45(f, Ng=Ng, Nr=runLength,ignore_zero=True)
    mat90 = my_glrlm_90(f, Ng=Ng, Nr=runLength,ignore_zero=True)
    mat135 = my_glrlm_135(f, Ng=Ng, Nr=runLength,ignore_zero=True)
    mat = np.dstack((mat0, mat45, mat90, mat135))
    return mat


def glrlm_features(f,Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    features : numpy ndarray
        1)Short Run Emphasis, 2)Long Run Emphasis, 3)Gray Level
        Non-Uniformity/Gray Level Distribution, 4)Run Length
        Non-Uniformity/Run Length Distribution, 5)Run Percentage,
        6)Low Gray Level Run Emphasis, 7)High Gray Level Run Emphasis,
        8)Short Low Gray Level Emphasis, 9)Short Run High Gray Level
        Emphasis, 10)Long Run Low Gray Level Emphasis, 11)Long Run
        High Gray Level Emphasis.
    labels : list
        Labels of features.
    '''


    labels = ["GLRLM_ShortRunEmphasis",
              "GLRLM_LongRunEmphasis",
              "GLRLM_GrayLevelNo-Uniformity",
              "GLRLM_RunLengthNonUniformity",
              "GLRLM_RunPercentage",
              "GLRLM_LowGrayLevelRunEmphasis",
              "GLRLM_HighGrayLevelRunEmphasis",
              "GLRLM_ShortLowGrayLevelEmphasis",
              "GLRLM_ShortRunHighGrayLevelEmphasis",
              "GLRLM_LongRunLowGrayLevelEmphasis",
              "GLRLM_LongRunHighGrayLevelEmphasis"]

    rlmatrix = my_glrlm(f,Ng)

    I, J = _calculate_ij(rlmatrix)
    S = _calculate_s(rlmatrix)
    G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
    R = np.apply_over_axes(np.sum, rlmatrix, axes=0)

    features = np.zeros(11,np.double)
    features[0] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[1] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[2] = ((np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0])/S).mean()
    features[3] = ((np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0])/S).mean()

    gray_level, run_length,_ = rlmatrix.shape
    num_voxels = gray_level * run_length
    features[4] = (S/num_voxels).mean()

    features[5]= ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[6] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[7] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()

    temp = _apply_over_degree(np.multiply, rlmatrix, (I*I))
    features[8] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()

    temp = _apply_over_degree(np.multiply, rlmatrix, (J*J))
    features[9] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[10] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()

    return features, labels


if __name__=="__main__":

    from pyfeats.textural.glrlm import glrlm
    import cv2
    import time
    from auxiliary.moment_preprocess import denoise


    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'

    img=cv2.imread(org_path%(1,2),0)
    img=denoise(img)
    img=255-img
    glrmat=my_glrlm(img)
    glrmat1=glrlm(img,np.ones(img.shape))
    if not (glrmat1!=glrmat).sum():
        print("正确的")

    

    # 效率对比
    # start=time.perf_counter()
    # for i in range(1,21):
    #     img=cv2.imread(org_path%(1,i),0)
    #     img=denoise(img)
    #     img=255-img
    #     img=np.round(img/255*8).astype(np.uint)
    #     glrmat=my_glrlm(img)
    # end=time.perf_counter()
    # print(end-start)
    #
    # start=time.perf_counter()
    # for i in range(1,21):
    #     img=cv2.imread(org_path%(1,i),0)
    #     img=denoise(img)
    #     img=255-img
    #     img=np.round(img/255*8).astype(np.uint)
    #     glrmat=glrlm(img,np.ones(img.shape))
    # end=time.perf_counter()
    # print(end-start)

    # test_arr=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])
    # mat1=my_glrlm_0(test_arr,6,5)