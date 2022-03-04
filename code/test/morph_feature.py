import cv2
import matplotlib.pyplot as plt

img1=cv2.imread(r'E:\\material\\signature\\signatures\\full_org\\original_22_14.png',0)
img2=cv2.imread(r'E:\\material\\signature\\signatures\\full_forg\\forgeries_22_4.png',0)
plt.subplot(121)
img1=255-img1
th1,re1=cv2.threshold(img1,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
img1[img1<th1]=0
img2=255-img2
th2,re2=cv2.threshold(img2,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
img2[img2<th2]=0
plt.figure()
plt.subplot(121)
plt.imshow(img1,cmap='gray')
plt.title('genuine')
plt.subplot(122)
plt.imshow(img2,cmap='gray')
plt.title('forge')

rect_k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
open1=cv2.morphologyEx(img1,cv2.MORPH_OPEN,rect_k)
open2=cv2.morphologyEx(img2,cv2.MORPH_OPEN,rect_k)
plt.figure()
plt.subplot(121)
plt.imshow(open1,cmap='gray')
plt.title('genuine(3*3 rect)')
plt.subplot(122)
plt.imshow(open2,cmap='gray')
plt.title('forge(3*3 rect)')

rect_x=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
open3=cv2.morphologyEx(img1,cv2.MORPH_OPEN,rect_x)
open4=cv2.morphologyEx(img2,cv2.MORPH_OPEN,rect_x)
plt.figure()
plt.subplot(121)
plt.imshow(open3,cmap='gray')
plt.title('genuine(3*3 cross)')
plt.subplot(122)
plt.imshow(open4,cmap='gray')
plt.title('forge(3*3 cross)')

center1=(int(img1.shape[0]/2),int(img1.shape[1]/2))
rm=cv2.getRotationMatrix2D(center1,-23,1)
rot_img1=cv2.warpAffine(img1,rm,(img1.shape[1],img1.shape[0]))
open5=cv2.morphologyEx(rot_img1,cv2.MORPH_OPEN,rect_x)
center2=(int(img2.shape[0]/2),int(img2.shape[1]/2))
rm=cv2.getRotationMatrix2D(center2,-30,1)
rot_img2=cv2.warpAffine(img2,rm,(img2.shape[1],img2.shape[0]))
open6=cv2.morphologyEx(rot_img2,cv2.MORPH_OPEN,rect_x)
plt.figure()
plt.subplot(121)
plt.imshow(open5,cmap='gray')
plt.title('genuine(3*3 cross rot)')
plt.subplot(122)
plt.imshow(open6,cmap='gray')
plt.title('forge(3*3 cross rot)')


ad_vec=[]
for i in [open1,open2,open3,open4,open5,open6]:
    ad_vec.append(temp_process(i,820,890))
ad_test=tf.data.Dataset.from_tensor_slices(ad_vec)
ad_vec=net.backbone.predict(ad_test.batch(6))
test_vec1=test_vec.copy()
test_vec1[266,:]=ad_vec[4,:]
test_vec1[247,:]=ad_vec[5,:]
feature_embedded1=tsne.fit_transform(test_vec1)
feature_embedded1=pd.DataFrame(feature_embedded1,columns=['dim_1','dim_2'])
label=pd.DataFrame(test_label,columns=['User','Authenticity'])
plt.figure()
import seaborn as sns
sns.scatterplot(data=pd.concat([feature_embedded1,label],axis=1),x='dim_1',y='dim_2',hue='User',style='Authenticity',palette='deep')
plt.scatter(feature_embedded1.loc[266,'dim_1'],feature_embedded1.loc[266,'dim_2'],marker='d',color='r',s=10)
plt.scatter(feature_embedded1.loc[247,'dim_1'],feature_embedded1.loc[247,'dim_2'],marker='d',color='b',s=10)