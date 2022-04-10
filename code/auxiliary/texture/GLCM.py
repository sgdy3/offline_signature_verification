# -*- coding: utf-8 -*-
# ---
# @File: GLCM.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/4/9
# Describe: 算了，还是自己实现一遍GLCM吧，pyfeats里的自定义化的程度太低了
# ---

'''
----------2022/4/10---------
GLCM不用显式注明Ng,内部调用了mahotas包，它会自动检测最大的灰度值作为Ng
所以这么想的话，是不是我也应该在代码里隐式计算Ng，这样肯定可以极大的提高计算速率

'''

import numpy as np
import mahotas

def glcm_features(f, ignore_zeros=True):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    ignore_zeros : int, optional
        Ignore zeros in image f. The default is True.

    Returns
    -------
    features_mean : numpy ndarray
        Haralick's 1)Angular Second Moment, 2)Contrast,
        3)Correlation, 4)Sum of Squares: Variance, 5)Inverse
        Difference Moment 6)Sum Average, 7)Sum Variance, 8)Sum
        Entropy, 9)Entropy, 10)Difference Variance, 11)Difference
        Entropy, 12)Information Measure of Correlation 1,
        13)Information Measure of Correlation 2, 14)Maximal
        Correlation Coefficient, mean
    features_range : numpy ndarray
        Haralick's features, same as before but range
    labels_mean : list
        Labels of features_mean.
    labels_range: list
        Labels of features_range.
    '''

    # 1) Labels
    labels = ["GLCM_ASM", "GLCM_Contrast", "GLCM_Correlation",
              "GLCM_SumOfSquaresVariance", "GLCM_InverseDifferenceMoment",
              "GLCM_SumAverage", "GLCM_SumVariance", "GLCM_SumEntropy",
              "GLCM_Entropy", "GLCM_DifferenceVariance",
              "GLCM_DifferenceEntropy", "GLCM_Information1",
              "GLCM_Information2", "GLCM_MaximalCorrelationCoefficient"]
    labels_mean = [label + "_Mean" for label in labels]
    labels_range = [label + "_Range" for label in labels]

    # 2) Parameters
    f = f.astype(np.uint8)

    # 3) Calculate Features: Mean and Range
    # 在计算矩阵的coccurence函数里可以看出这里的Ng是自动选取f中最大的，不用再进行设置了。
    features = mahotas.features.haralick(f,
                                         ignore_zeros=ignore_zeros,
                                         compute_14th_feature=True,
                                         return_mean_ptp=True)
    features_mean = features[0:14]
    features_range = features[14:]

    return features_mean, features_range, labels_mean, labels_range
