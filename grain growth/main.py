# -*- coding: utf-8 -*-
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
from skimage.measure import regionprops,label

x_data = np.load('xData.npy')
y_data = np.load('yData.npy')
print("Load data done!")  

load_weights = True

delta_t = [1,2,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,26,26,27,28,30,3,10,21,25,29]
delta_t = [element * 4 for element in delta_t]
deltaT = np.ndarray((600*95,),dtype = np.float16) 
count = 0
for i in range(0,600):
    for j in range(0,95):
        deltaT[count] = delta_t[int(i/20)]
        count = count + 1
deltaT = deltaT/60.0

nx = 128
ny = 128

x_data2 = np.ndarray((600*95,nx,ny),dtype = np.float32) 
y_data2 = np.ndarray((600*95,nx,ny),dtype = np.float32) 
count = 0
for i in range(0,600):
    for j in range(0,95):
        x_data2[count,:,:] = x_data[i,j,:,:]
        y_data2[count,:,:] = y_data[i,j,:,:]
        count = count + 1                    


x_train = x_data2[0*95:500*95,:,:]
x_test = x_data2[500*95:600*95,:,:]
y_train = y_data2[0*95:500*95,:,:]
y_test = y_data2[500*95:600*95,:,:]
deltaT_train = deltaT[0*95:500*95]
deltaT_test = deltaT[500*95:600*95]

nx2 = 128
ny2 = 128
x_train = np.reshape(x_train, (len(x_train), nx2, ny2, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), nx2, ny2, 1))  # adapt this if using `channels_first` image data format  
y_train = np.reshape(y_train, (len(y_train), nx2, ny2, 1))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), nx2, ny2, 1))  # adapt this if using `channels_first` image data format  
deltaT_train = np.reshape(deltaT_train, (len(deltaT_train), 1))
deltaT_test = np.reshape(deltaT_test, (len(deltaT_test), 1))


  
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='binary_crossentropy')

if load_weights == True:
    MICNN.load_weights("weights_yNet.h5")
else: 
    model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
    MICNN.fit([x_train,deltaT_train], y_train,
                    epochs=100,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([x_test,deltaT_test], y_test))

decoded_imgs = MICNN.predict([x_test,deltaT_test])

####################5 random testing reulsts####
for i in range(0,5):
    rand1 = np.random.rand()*100*95
    t_no = int(rand1)
    #Display intinal grain structure
    ax = plt.imshow(x_test[t_no].reshape(nx2, ny2),cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testno'+str(i+1)+'_1.jpg', dpi=150, bbox_inches = "tight")
    
    # display evolved grain structure (ground truth)
    ax = plt.imshow(y_test[t_no].reshape(nx2, ny2),cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testno'+str(i+1)+'_2.jpg', dpi=150, bbox_inches = "tight")
    # display eovolved structure by yNet prediction
    ax = plt.imshow(decoded_imgs[t_no].reshape(nx2, ny2),cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testno'+str(i+1)+'_3.jpg', dpi=150, bbox_inches = "tight")
    
    print(deltaT_test[t_no,0]*15)
    
    #Display error
    xP = decoded_imgs[t_no].reshape(nx2, ny2)
    xG = y_test[t_no].reshape(nx2, ny2)
    AError = np.absolute(xP - xG) 
    ax = plt.imshow(AError[:,:],cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testno'+str(i+1)+'_4.jpg', dpi=150, bbox_inches = "tight")
    plt.close()



##############Calculate RMSE for 100*95 cases##########################
mMSE = np.zeros(100*95,)
mRMSE = np.zeros(100*95,)
for i in range(0,100*95):
    x_test_0_2D_pre = np.zeros((nx2,ny2))
    x_test_0_2D_GT = np.zeros((nx2,ny2))
    t_no = i
    x_test_0_2D_pre = decoded_imgs[t_no].reshape(nx2, ny2)
    x_test_0_2D_GT = y_test[t_no].reshape(nx2, ny2)
    errorMSE = np.mean((x_test_0_2D_pre - x_test_0_2D_GT)**2)
    mMSE[i] = errorMSE
    mRMSE[i] = np.sqrt(errorMSE)

print('Average RMSE:',np.mean(mRMSE))



