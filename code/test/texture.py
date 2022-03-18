import numpy as np
import cv2
import pyfeats
from skimage.feature import greycomatrix, greycoprops
from itertools import groupby
import pyfeats as pf
from auxiliary.preprocessing import hafemann_preprocess



def GLCM(gray):


    # get a gray level co-occurrence matrix (GLCM)
    # parameters：the matrix of gray image，distance，direction，gray level，symmetric or not，standarzation or not
    glcm = greycomatrix(gray, [1],[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                        256, symmetric = True, normed = True)

    print(glcm.shape); print("===============================\n")

    #获取共生矩阵的统计值.
    feature=[]
    for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
        # 对比度，相异性，反方差矩阵，能量，相关系数，角二阶矩
        temp = greycoprops(glcm, prop)
        feature.append(temp)

        print(prop, temp)
        # print(prop + "_mean: ", np.mean(temp))
        # print(prop + "_std:", np.std(temp, ddof = 1));
        print( "==============================\n")
    return feature


def getGrayLevelRumatrix( array, theta):
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

org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
test_arr=np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3],[1,2,3,2]])
test_img=cv2.imread(org_path%(1,1),0)
test_img=hafemann_preprocess(test_img,820,890)
features_mean, features_range, labels_mean, labels_range=pf.glcm_features(test_img,True)  # feature_range为统计量max-min的值
glrlm_feature,labels=pf.glrlm_features(test_img,None)
