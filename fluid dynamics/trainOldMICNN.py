# -*- coding: utf-8 -*-
#Train the conventional flattening based multi-input CNN
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

MICNN = oldMICNN(nx2,ny2)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='mean_squared_error')
dot_img_file = 'architecture_oldMICNN.png'
tf.keras.utils.plot_model(MICNN, to_file=dot_img_file, show_shapes=True)

model_checkpoint = ModelCheckpoint('weights_oldMICNN.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
MICNN.fit([x_train,rey_train], y_train,
                epochs=100,
                batch_size=2,
                shuffle=True,
                callbacks=[model_checkpoint],
                validation_data=([x_test,rey_test], y_test))

