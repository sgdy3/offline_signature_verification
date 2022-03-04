from keras.layers import LSTM
from keras.models import  Sequential,Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lat_file=r'E:\material\TJport\discuss\final\sd_lat.csv'
lng_file=r'E:\material\TJport\discuss\final\sd_lng.csv'
test_file=r'E:\material\TJport\discuss\final\x_test.csv'
pred_file=r'E:\material\TJport\discuss\final\x_pred.csv'

lat=pd.read_csv(lat_file)
lng=pd.read_csv(lng_file)
lat_test=lat.loc[:,['0','1','2','3']].values
lat_test=np.expand_dims(lat_test,2)
lng_test=lng.loc[:,['0','1','2','3']].values
lng_test=np.expand_dims(lng_test,2)
x_train=np.concatenate([lat_test,lng_test],axis=2)
x_mean=x_train.mean()
x_std=x_train.std()
x_train=(x_train-x_train.mean())/x_train.std()
y_train=np.vstack([lat.loc[:,'4'].values,lng.loc[:,'4'].values]).T
y_mean=y_train.mean()
y_std=y_train.std()
y_train=(y_train-y_train.mean())/y_train.std()


model=Sequential()
model.add(LSTM(units=2, input_shape=(4, 2), return_sequences=False))
model.compile(optimizer='Adam',loss='mean_squared_error',metrics=['mean_squared_error',])
loss=model.fit(x_train,y_train,batch_size=32,epochs=50)
plt.plot(loss.epoch,loss.history['loss'])

x_test=pd.read_csv(test_file)
x_test=x_test.values
lat_test=x_test[0:x_test.shape[0]:2,:]
lng_test=x_test[1:x_test.shape[0]:2,:]
x_test=np.concatenate([lat_test.reshape((-1,4,1)),lng_test.reshape((-1,4,1))],axis=2)
x_test=(x_test-x_mean)/x_std # 归一化
y_pred=model.predict(x_test)
y_pred=y_pred*y_std+y_mean
y_pred=pd.DataFrame(y_pred,columns=['lat','lng'])
y_pred.to_csv(pred_file,index=False)

'''
多步预测20，30min
'''
lat_test=lat_test[:,1:4]
lng_test=lng_test[:,1:4]
lat_test=np.hstack([lat_test,y_pred['lat'].values.reshape(-1,1)])
lng_test=np.hstack([lng_test,y_pred['lng'].values.reshape(-1,1)])