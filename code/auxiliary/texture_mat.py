# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/13
# Describe: pyfeats的设置过于僵硬，许多设置不能直接在feature接口中直接调用，重新实现一遍并进行纠错
# ---


import numpy as np
from skimage import measure
from scipy import signal


def glszm(f,mask,Ng=256,ignore_zero=True,connectivity=2):
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

def glszm_features(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.

    Returns
    -------
    features : numpy ndarray
        1)Small Zone Emphasis, 2)Large Zone Emphasis,
        3)Gray Level Nonuniformity, 4)Zone Size Nonuniformit,
        5)Zone Percentage, 6)Low Gra yLeveL Zone Emphasis,
        7)High Gray Level Zone Emphasis, 8)Small Zone Low Gray
        Level Emphasis, 9)Small Zone High Gray LeveL Emphasis,
        10)Large Zone Lo wGray Level Emphassis, 11)Large Zone High
        Gray Level Emphasis, 12)Gray Level Variance,
        13)Zone Size Variance, 14)Zone Size Entropy.
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)

    labels = ['GLSZM_SmallZoneEmphasis', 'GLSZM_LargeZoneEmphasis',
              'GLSZM_GrayLevelNonuniformity', 'GLSZM_ZoneSizeNonuniformity',
              'GLSZM_ZonePercentage', 'GLSZM_LowGrayLeveLZoneEmphasis',
              'GLSZM_HighGrayLevelZoneEmphasis', 'GLSZM_SmallZoneLowGrayLevelEmphasis',
              'GLSZM_SmallZoneHighGrayLevelEmphasis', 'GLSZM_LargeZoneLowGrayLevelEmphassis',
              'GLSZM_LargeZoneHighGrayLevelEmphasis', 'GLSZM_GrayLevelVariance',
              'GLSZM_ZoneSizeVariance','GLSZM_ZoneSizeEntropy']

    P = glszm(f, mask)
    # FIXME
    #idx = np.argwhere(np.all(P[..., :] == 0, axis=0))
    #P = np.delete(P, idx, axis=1)

    p = P / P.sum()

    Ng, Ns = p.shape
    pg = np.sum(p,1) # Gray-Level Sum [Ng x 1]
    ps = np.sum(p,0) # Zone-Size Sum  [Ns x 1]
    jvector = np.arange(1,Ns+1)
    ivector = np.arange(1,Ng+1)
    Nz = np.sum(P, (0,1))
    Np = np.sum(ps * jvector, 0)
    [imat,jmat] = np.meshgrid(jvector,ivector)

    features = np.zeros(14, np.double)
    features[0] = np.dot(ps,((1/(jvector+1e-16))**2))
    features[1]= np.dot(ps, jvector ** 2)
    features[2] = (pg**2).sum()
    features[3] = (ps**2).sum()
    features[4] = Nz / Np
    features[5] = np.dot(pg, 1/(ivector+1e-16)**2)
    features[6] = np.dot(pg, ivector ** 2)
    features[7] =  np.multiply(p,
                               np.multiply(1/(jmat+1e-16)**2,1/(imat+1e-16)**2)).sum()
    features[8] = np.multiply(p,
                              np.multiply(jmat**2,1/(imat+1e-16)**2)).sum()
    features[9] = np.multiply(p,
                              np.multiply(1/(jmat+1e-16)**2,imat**2)).sum()
    features[10] = np.multiply(p, np.multiply(jmat**2,imat**2)).sum()
    meang = np.dot(pg,ivector)/(Ng*Ns)
    features[11] = ((np.multiply(p, jmat) - meang) ** 2).sum() / (Ng*Ns)
    means = np.dot(ps,jvector)/(Ng*Ns)
    features[12] = ((np.multiply(p, imat) - means) ** 2).sum() / (Ng*Ns)
    features[13] = np.multiply(p, np.log2(p+1e-16)).sum()

    return features, labels



def _image_xor(f):
    # Turn "0" to "1" and vice versa: XOR with image consisting of "1"s
    f = f.astype(np.uint8)
    mask = np.ones(f.shape, np.uint8)
    out = np.zeros(f.shape, np.uint8)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i,j] = f[i,j] ^ mask[i,j]
    return out


def ngtdm(f, mask, d, Ng=256):
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
            if conv_mask[x,y] > 0:
                index = f[x,y].astype('i')
                S[index] = S[index] + diff[x,y]
                N[index] += 1

    R = sum(N)

    return S, N, R


def ngtdm_features(f, mask, d=1):
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
    Ng = 256

    # 3) Calculate NGTDM
    S, N, R = ngtdm(f, mask, d, Ng)

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
    test_arr=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])
    g_m=glszm(test_arr,np.ones(test_arr.shape),6)
    test_arr=np.array([[1,2,5,2],[3,5,1,3],[1,3,5,5],[3,1,1,1]])
    temp=np.ones(test_arr.shape)
    # temp[1,2]=0
    ngtdmat=ngtdm(test_arr,temp,1,6)