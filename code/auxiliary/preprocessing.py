import cv2
import numpy as np
from itertools import combinations
import pickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage


'''
2022.03.19
1. 目前可以确保此处的预处理方法是完全正确的，和hafemann提出的方法一致，唯一有待争议的地方是是否需要进行标准化
在代码中并没有找到相关部分。
2. hafemann的预处理方法看来效果并不好，因为他先将图片复制到了一个较大的模板上，又从
'''


def load_image(file_name):
    img = tf.io.read_file(file_name, 'rb')  # 读取图片
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    return img



def hafemann_preprocess(img,ext_h,ext_w,dst_size=(150,220),img_size=(170, 242)):
    img=np.squeeze(img)
    centered_img=centered(img,ext_h,ext_w)  # 将图像的质心和幕布中心对齐
    inverted_img=255-centered_img  # 反色，背景为0，签名为255-x
    resized_img=resize_img(inverted_img,img_size)
    croped_img=crop_center(resized_img,dst_size)
    croped_img=croped_img.astype(np.uint8)
    return croped_img


def centered(img,ext_h,ext_w):
    # 先用高斯滤波去除图中的小组件
    radius=2
    blurred_img=ndimage.gaussian_filter(img,radius)
    # 求取质心
    threshold,binarized_img=cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    r,c=np.where(binarized_img==0) # 有笔画的位置
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())
    cropped = img[r.min(): r.max(), c.min(): c.max()] # 包围笔画的矩形框
    # 将笔画框的质心和幕布的中心对齐
    img_r, img_c = cropped.shape
    r_start = ext_h // 2 - r_center
    c_start = ext_w // 2 - c_center  # 求取对齐时笔画框左上角在幕布上的位置

    if img_r > ext_h: # 签名高度比幕布高度大
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_r - ext_h
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + ext_h, :]
        img_r = ext_h
    else:
        # 防止对齐后的笔画框超出幕布范围
        extra_r = (r_start + img_r) - ext_h
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0: # 如果要对齐左上角会超出幕布范围就放弃对齐
            r_start = 0

    if img_c > ext_w:
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_c - ext_w
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + ext_w]
        img_c = ext_w
    else:
        extra_c = (c_start + img_c) - ext_w
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0
    normalized_image = np.ones((ext_h, ext_w), dtype=np.uint8) * 255
    normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped
    normalized_image[normalized_image > threshold] = 255 # 用大津法确定的阈值去背景
    return normalized_image

def resize_img(img,new_size):
    # 先将一边缩小到指定大小，保持比例不变缩小另一边，剪切。
    dst_h,dst_w=new_size
    h_scale=float(img.shape[0])/dst_h
    w_scale=float(img.shape[1])/dst_w
    if w_scale>h_scale:
        resized_height=dst_h
        resized_width=int(round(img.shape[1]/h_scale))
    else:
        resized_width=dst_w
        resized_height=int(round(img.shape[0]/w_scale))
    img=cv2.resize(img.astype(np.float32),(resized_width,resized_height))
    if w_scale>h_scale:
        start = int(round((resized_width-dst_w)/2.0))
        return img[:, start:start+dst_w]
    else:
        start = int(round((resized_height-dst_h)/2.0))
        return img[start:start+dst_h, :]

def crop_center(img, input_shape):
    img_shape = img.shape
    start_y = (img_shape[0] - input_shape[0]) // 2
    start_x = (img_shape[1] - input_shape[1]) // 2
    cropped = img[start_y: start_y + input_shape[0], start_x:start_x + input_shape[1]]
    return cropped


def preprocess(img, ext_h, ext_w,dst_h=150,dst_w=220):
    # 改进的预处理方法，不再是直接质心对齐，保持纵横比不变的情况下先将一条边扩大到指定大小，再resize到网络输入
    h,w,_=img.shape
    scale=min(ext_h / h, ext_w / w)
    nh=int(scale*h)
    nw=int(scale*w)
    img=tf.image.resize(img,[nh,nw])
    pad_row1=int((ext_h - img.shape[0]) / 2)
    pad_row2= (ext_h - img.shape[0]) - pad_row1
    pad_col1=int((ext_w - img.shape[1]) / 2)
    pad_col2= (ext_w - img.shape[1]) - pad_col1
    img=tf.pad(img,[[pad_row1,pad_row2],[pad_col1,pad_col2],[0,0]],constant_values=255)
    img=tf.image.resize(img,[dst_h,dst_w])
    #img=otsu(img.numpy())
    threshold=cv2.threshold(img.numpy().astype(np.uint8),0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)[0]#大津法确定阈值
    img=img.numpy()
    # img[img>threshold]=255 # 大于阈值的变为白色
    img[img<threshold]=255-img[img<threshold]
    img=255-img # 背景黑色，笔迹白色
    img=img.astype(np.uint8)
    return img





def img_var(img_data):
    # 提取所有先前处理好的图片后计算方差，很慢，特别慢，怀疑是阈值化那一步里使用自制大津法的效率过低
    temp=np.zeros((155,220,1))
    for i in img_data.as_numpy_iterator():
        temp=np.concatenate([temp,i[0]],axis=2)
    return np.sqrt(temp.var) #2640张图片的标准差为17.87


if __name__=="__main__":
    path=r'E:\\temp\\some_signature.png'
    img1 = cv2.imread(path, 0)  # 读取图片
    img1=np.squeeze(img1)
    normalized = 255 - centered(img1, 952, 1360)
    resized = resize_img(normalized, (170, 242))
    cropped = crop_center(resized, (150,220))

    f, ax = plt.subplots(4,1, figsize=(6,15))
    ax[0].imshow(img1, cmap='Greys_r')
    ax[1].imshow(normalized)
    ax[2].imshow(resized)
    ax[3].imshow(cropped)

    ax[0].set_title('Original')
    ax[1].set_title('Background removed/centered')
    ax[2].set_title('Resized')
    ax[3].set_title('Cropped center of the image')

    # after1=hafemann_preprocess(img1,730,1042)
    # after2=preprocess(img1,730,1042)
    # temp1=after1.copy()
    # temp2=after2.copy()
    # temp2=np.squeeze(temp2)
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(np.squeeze(img1.numpy()),cmap='gray')
    # plt.subplot(132)
    # plt.imshow(temp1,cmap='gray')
    # plt.subplot(133)
    # plt.imshow(temp2,cmap='gray')