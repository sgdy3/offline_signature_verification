# -*- coding: utf-8 -*-
# ---
# @File: NGTDM.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/4/10
# Describe: 修改后的NGTDM求解
# ---


'''
-----------2022/04/10-------------
自己写的NGTDM算法比原来的速度要快，无论是否做color reduce;
但这里的ngtdm也不是pyfeats里的原始实现，而是保持了逻辑不变，对原始实现的mask机制进行了修改的版本
'''

import numpy as np
from scipy import signal


def _image_xor(f):
    # Turn "0" to "1" and vice versa: XOR with image consisting of "1"s
    f = f.astype(np.uint8)
    mask = np.ones(f.shape, np.uint8)
    out = np.zeros(f.shape, np.uint8)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i,j] = f[i,j] ^ mask[i,j]
    return out


def my_ngtdm(f, mask, d, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    d : int, optional
        Distance for NGTDM. Default is 1.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    S : numpy ndarray
    N : numpy ndarray
    R : numpy ndarray
    '''

    f = f.astype(np.double)
    oneskernel = np.ones((2*d+1,2*d+1))
    kernel = oneskernel.copy()
    kernel[d,d] = 0

    # Get complementary mask
    mask_c = _image_xor(mask)

    # Find which pixels are inside mask for convolution
    conv_mask = signal.convolve2d(mask_c,oneskernel,'same')
    conv_mask = abs(np.sign(conv_mask)-1)

    # Calculate abs diff between actual and neighborhood
    cov_mask2=signal.convolve2d(mask,kernel,'same')
    B = signal.convolve2d(f,kernel,'same')
    B = B/cov_mask2

    diff = abs(f-B)
    diff=diff*mask

    # Construct NGTDM matrix
    S = np.zeros(Ng,np.double)
    N = np.zeros(Ng,np.double)
    for i in range(Ng):
        S[i]=diff[np.where(f==i)].sum()
        N[i]=mask[np.where(f==i)].sum()

    R = sum(N)

    return S, N, R

def ngtdm_v1(f, mask, d, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    d : int, optional
        Distance for NGTDM. Default is 1.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    S : numpy ndarray
    N : numpy ndarray
    R : numpy ndarray
    '''

    f = f.astype(np.double)
    N1, N2 = f.shape
    oneskernel = np.ones((2*d+1,2*d+1))
    kernel = oneskernel.copy()
    kernel[d,d] = 0
    W = (2*d + 1)**2

    # Get complementary mask
    mask_c = _image_xor(mask)

    # Find which pixels are inside mask for convolution
    conv_mask = signal.convolve2d(mask_c,oneskernel,'same')
    conv_mask = abs(np.sign(conv_mask)-1)

    # Calculate abs diff between actual and neighborhood
    cov_mask2=signal.convolve2d(mask,kernel,'same')
    B = signal.convolve2d(f,kernel,'same')
    B = B/cov_mask2

    diff = abs(f-B)

    # Construct NGTDM matrix
    S = np.zeros(Ng,np.double)
    N = np.zeros(Ng,np.double)
    for x in range(N1):
        for y in range(N2):
            if mask[x,y] > 0:
                index = f[x,y].astype('i')
                S[index] = S[index] + diff[x,y]
                N[index] += 1

    R = sum(N)

    return S, N, R

def ngtdm_features(f, mask, Ng=256,d=1):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    d : int, optional
        Distance for NGTDM. Default is 1.

    Returns
    -------
    features : numpy ndarray
        1)Coarseness, 2)Contrast, 3)Busyness, 4)Complexity, 5)Strength.
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)

    # 1) Labels
    labels = ["NGTDM_Coarseness","NGTDM_Contrast","NGTDM_Busyness",
              "NGTDM_Complexity","NGTDM_Strngth"]

    # 2) Parameters
    f  = f.astype(np.uint8)
    mask = mask.astype(np.uint8)


    # 3) Calculate NGTDM
    S, N, R = my_ngtdm(f, mask, Ng,d)

    # 4) Calculate Features
    features = np.zeros(5,np.double)
    Ni, Nj = np.meshgrid(N,N)
    Si, Sj = np.meshgrid(S,S)
    i, j = np.meshgrid(np.arange(Ng),np.arange(Ng))
    ilessjsq = ((i-j)**2).astype(np.double)
    Ni = np.multiply(Ni,abs(np.sign(Nj)))
    Nj = np.multiply(Nj,abs(np.sign(Ni)))
    features[0] = R*R / sum(np.multiply(N,S))
    features[1] = sum(S)*sum(sum(np.multiply(np.multiply(Ni,Nj),ilessjsq)))/R**3/Ng/(Ng-1)
    temp = np.multiply(i,Ni) - np.multiply(j,Nj)
    features[2] = sum(np.multiply(N,S)) / sum(sum(abs(temp))) / R
    temp = np.multiply(Ni,Si) + np.multiply(Nj,Sj)
    temp2 = np.multiply(abs(i-j),temp)
    temp3 = np.divide(temp2,Ni+Nj+1e-16)
    features[3] = sum(sum(temp3)) / R
    features[4] = sum(sum(np.multiply(Ni+Nj,ilessjsq))) / (sum(S)+1e-16)

    return features, labels

if __name__=="__main__":

    import cv2
    import time
    from auxiliary.moment_preprocess import denoise
    from pyfeats.textural.ngtdm import ngtdm

    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'

    img=cv2.imread(org_path%(1,2),0)
    img=denoise(img)
    img=255-img
    ngtdmat=my_ngtdm(img,np.ones(img.shape),1)
    ngtdmat1=ngtdm(img,np.ones(img.shape),1)
    ngtdmat2=ngtdm_v1(img,np.ones(img.shape),1)
    # 这里不会输出正确的，因为自己实现的方法考虑了边界像素，pyfeats里的没有
    if not (ngtdmat1[0]!=ngtdmat[0]).sum():
        print("正确的")
    if (ngtdmat2[0]-ngtdmat[0]).sum()<1e-10:
        print("正确的")


    # 效率对比
    # start=time.perf_counter()
    # for i in range(1,21):
    #     img=cv2.imread(org_path%(1,i),0)
    #     img=denoise(img)
    #     img=255-img
    #     #img=np.round(img/255*7).astype(np.uint)
    #     ngtdmat=ngtdm_v1(img,np.ones(img.shape),1,256)
    # end=time.perf_counter()
    # print(end-start)
    #
    # start=time.perf_counter()
    # for i in range(1,21):
    #     img=cv2.imread(org_path%(1,i),0)
    #     img=denoise(img)
    #     img=255-img
    #     #img=np.round(img/255*7).astype(np.uint)
    #     ngtdmat1=my_ngtdm(img,np.ones(img.shape),1,256)
    # end=time.perf_counter()
    # print(end-start)

    # test_arr=np.array([[1,2,5,2],[3,5,1,3],[1,3,5,5],[3,1,1,1]])
    # temp=np.ones(test_arr.shape)
    # ngtdmat=ngtdm(test_arr,temp,1,6)
    # n=my_ngtdm(test_arr,temp,1,6)