# -*- coding: utf-8 -*-
#Training and testing of yNet
import os

import numpy as np
import matplotlib.pyplot as plt
from model import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime


n = 29  #number of Re sampling points
nframe = 160  #frames of each simulation

xData0 = np.load('xData.npy')
yData0 = np.load('yData.npy')
print('Data loaded!') 
rey = np.zeros(n*(nframe-1),)
Re = [0,1,2,4,5,6,7,9,10,11,12,13,14,15,16,18,19,21,22,23,25,26,27,28,3,8,17,20,24]  #First 24 for training, last 5 for testing
count = 0
for i in range(0,n):
    for j in range(0,(nframe-1)):
        rey[count] = Re[i]*3 + 75
        count = count + 1
rey = (rey- np.min(rey))/(np.max(rey)-np.min(rey))  
       
    
nx = 256
ny = 64
no = 0
x_data = xData0[:,0:256,0:64]
y_data = yData0[:,0:256,0:64]


x_train = x_data[0*(nframe-1):24*(nframe-1),:,:]
x_test = x_data[24*(nframe-1):n*(nframe-1),:,:]
y_train = y_data[0*(nframe-1):24*(nframe-1),:,:]
y_test = y_data[24*(nframe-1):n*(nframe-1),:,:]
rey_train = rey[0*(nframe-1):24*(nframe-1)]
rey_test = rey[24*(nframe-1):n*(nframe-1)]
 

print(x_train.shape)
print(y_train.shape)
print(rey_train.shape)
print(x_test.shape)
print(y_test.shape)
print(rey_test.shape)


nx2 = 256
ny2 = 64
x_train = np.reshape(x_train, (len(x_train), nx2, ny2, 1))  
x_test = np.reshape(x_test, (len(x_test), nx2, ny2, 1))  
y_train = np.reshape(y_train, (len(y_train), nx2, ny2, 1))  
y_test = np.reshape(y_test, (len(y_test), nx2, ny2, 1))
rey_train = np.reshape(rey_train, (len(rey_train), 1))
rey_test = np.reshape(rey_test, (len(rey_test), 1))

MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='mean_squared_error')
dot_img_file = 'architecture_yNet.png'
tf.keras.utils.plot_model(MICNN, to_file=dot_img_file, show_shapes=True)

model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
MICNN.fit([x_train,rey_train], y_train,
                epochs=100,
                batch_size=2,
                shuffle=True,
                callbacks=[model_checkpoint],
                validation_data=([x_test,rey_test], y_test))

decoded_imgs = MICNN.predict([x_test,rey_test])


########Testing of yNet under 5 unused Re conditions#################
vMin = 0.3
vMax = 0.7
for i in range(0,5):
    
###Save five random testing results.
    test_frame_no = np.random.rand()*(nframe-1)
    t_no = int(test_frame_no)+(nframe-1)*i
    print(rey_test[t_no,0]*(159-75)+75)    

#    Display input flow field
    ax = plt.imshow(np.rot90(x_test[t_no].reshape(nx2, ny2)),cmap = 'seismic', vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    
#    Display evolved field: ground truth
    ax = plt.imshow(np.rot90(y_test[t_no].reshape(nx2, ny2)),cmap = 'seismic',vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display evolved field: MICNN prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx2, ny2)),cmap = 'seismic',vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    
    #Display absolute error
    xP = decoded_imgs[t_no].reshape(nx2, ny2)
    xG = y_test[t_no].reshape(nx2, ny2)
    AError = ( (xP*352.15*2-352.15) - (xG*352.15*2-352.15)  )
    ax = plt.imshow(np.rot90(AError[:,:]),cmap = 'seismic', vmin = -3, vmax = 3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testCylno'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()

####Calculate MSE and RMSE for all 5x159 testing cases
MSE_norm = 0.0
MSE = 0.0
RMSE = 0.0
for i in range(0,5*(nframe-1)):
    xP = decoded_imgs[i].reshape(nx2, ny2)
    xG = y_test[i].reshape(nx2, ny2)
    MSE_norm = np.mean((xP- xG)**2) + MSE_norm
    MSE = np.mean((xP*352.15*2 - xG*352.15*2)**2) + MSE
MSE_norm = MSE_norm/(5*(nframe-1))
MSE = MSE/(5*(nframe-1))
RMSE = np.sqrt(MSE)
print('MSE_norm',MSE_norm, 'MSE:', MSE,'rmse:',RMSE)



    


    

    
    


