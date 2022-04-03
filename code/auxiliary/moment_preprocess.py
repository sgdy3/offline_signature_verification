# -*- coding: utf-8 -*-
# ---
# @File: moment_preprocess.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/19
# Descibe:  Another way to preprocess the offline signature except it applied bty hafemann
# ---

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def moment_preprocess(img,dst_w=220,dst_h=150):
    normalized_img = denoise(img)
    inverted_img = 255 - normalized_img
    resized_img=resize_img(inverted_img)
    cropped_img=crop_center(resized_img,(dst_h,dst_w))
    return cropped_img


def denoise(img,reigon_mask=False):
    radius=2
    blurred_img=ndimage.gaussian_filter(img,radius)
    threshold, binarized_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 大于OTSU的为噪声，设置为255
    r, c = np.where(binarized_img == 255)  # 有笔画的位置
    img[img>threshold]=255
    if reigon_mask:
        # 返回包含签名的最小矩形框
        cropped = img[r.min(): r.max(), c.min(): c.max()]
        return cropped
    else:
        return img


def resize_img(img, img_size=(150,220), K=2.5):
    dst_h,dst_w=img_size
    moments = cv2.moments(img, True)  # 计算图像的矩，不考虑灰度的变化，二值化图像01
    xc = moments['m10'] / moments['m00']
    yc = moments['m01'] / moments['m00']
    ratio = min(dst_w * np.sqrt(moments['m00']) / (2 * K * np.sqrt(moments['mu20'])),
                dst_h * np.sqrt(moments['m00']) / (2 * K * np.sqrt(moments['mu02'])))
    mat = np.array([[ratio, 0, -xc * ratio + (dst_w - 1) / 2], [0, ratio, -yc * ratio + (dst_h - 1) / 2]])
    trans_img = cv2.warpAffine(img, mat, (dst_w, dst_h),flags=cv2.INTER_LINEAR)
    return trans_img


def crop_center(img, input_shape):
    dst_h,dst_w = input_shape
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


if __name__ == "__main__":
    path = r'E:\\temp\\some_signature.png'
    img = cv2.imread(path, 0)
    normalized = 255-denoise(img)
    resized = resize_img(normalized)
    cropped= crop_center(resized,(150,220))

    f, ax = plt.subplots(4,1, figsize=(6,15))
    ax[0].imshow(img, cmap='Greys_r')
    ax[1].imshow(normalized)
    ax[2].imshow(resized)
    ax[3].imshow(cropped)


    ax[0].set_title('Original')
    ax[1].set_title('Background removed/centered')
    ax[2].set_title('Resized')
    ax[3].set_title('Cropped center of the image')
