# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/13
# Descibe: pyfeats的设置过于僵硬，许多设置不能直接在feature接口中直接调用，重新实现一遍
# ---


from pyfeats import textural
import numpy as np
from skimage import measure
from scipy import signal

def glszm(f, mask,Ng=256,ignore_zero=True):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.

    Returns
    -------
    GLSZM : numpy ndarray
        GLSZ Matrix.
    '''
    levels=np.arange(0,Ng)

    Ns = f.shape[0] * f.shape[1]  # maxsize of connected region
    GLSZM = np.zeros((Ng,Ns), np.double)

    temp = f.copy()
    for i in range(Ng):
        temp[f!=levels[i]] = 0
        temp[f==levels[i]] = 1
        connected_components = measure.label(temp, connectivity=2)
        connected_components = connected_components * mask
        nZone = len(np.unique(connected_components))
        for j in range(nZone):
            col = np.count_nonzero(connected_components==j)
            GLSZM[i,col-1] += 1  # 连通域size为1时对应第0列

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
    A = signal.convolve2d(f,kernel,'same') / (W-1)
    diff = abs(f-A)

    # Construct NGTDM matrix
    S = np.zeros(Ng,np.double)
    N = np.zeros(Ng,np.double)
    for x in range(d,(N1-d)):
        for y in range(d,(N2-d)):
            if conv_mask[x,y] > 0:
                index = f[x,y].astype('i')
                S[index] = S[index] + diff[x,y]
                N[index] += 1

    R = sum(N)

    return S, N, R



test_arr=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])
g_m=glszm(test_arr,np.ones(test_arr.shape),6)