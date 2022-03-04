import cv2 as cv
import tensorflow as tf
from code.auxiliary.preprocessing import preprocess
import numpy as np
import matplotlib.pyplot as plt

file_name1=r'E:\material\signature\signatures\full_org\original_1_3.png'
img1 = tf.io.read_file(file_name1, 'rb')
img1 = tf.image.decode_png(img1, channels=3)
img1 = tf.image.rgb_to_grayscale(img1)
surf=cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img1.numpy(),None)
temp_img=np.squeeze(img1.numpy())
pt=[i.pt for i in kp]
pt=np.array(pt)
loc=np.zeros((pt.shape[0],4))
loc[:,0]=pt[:,1]-2
loc[:,1]=pt[:,0]-2
loc[:,2]=pt[:,1]+3
loc[:,3]=pt[:,0]+3 # 囊括特征点周围3*3的领域,用检测出的size的话太大了
loc=loc.astype(int)
contours_map=np.zeros(temp_img.shape)
for i in range(50):
    pos=loc[i]
    contours_map[pos[0]:pos[2],pos[1]:pos[3]]=temp_img[pos[0]:pos[2],pos[1]:pos[3]]
contours_map=contours_map.astype(np.uint8)
plt.figure(1)
plt.imshow(contours_map)
plt.figure(2)
plt.imshow(temp_img)
img2 = cv.drawKeypoints(temp_img,kp,None,(255,0,0),4)
plt.figure(3)
plt.imshow(img2)
img1=preprocess(img1,820,890)