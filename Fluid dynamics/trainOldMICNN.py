# -*- coding: utf-8 -*-
#Train the conventional flattening based multi-input CNN
# -*- coding: utf-8 -*-
#Training and testing of the proposed gating-based yNet

import numpy as np
import matplotlib.pyplot as plt
from model import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


n = 30  #number of Re sampling points
nframe = 160  #frames of each simulation
nx = 256
ny = 64
valMin = -355.57
valMax = 355.57


##Load xData and yData
xData = np.load('xData.npy')
yData = np.load('yData.npy')
print('Data loaded!')
xData = xData[:,0:nx,0:ny] #257x65 -> 256x54
yData = yData[:,0:nx,0:ny] #257x65 -> 256x54


##Create reynolds number dataset (reData) based on the data-splitting scheme during data preprocessing
reData = np.zeros(n*(nframe-1),)
mRe = 75+np.array([0,1,2,4,5,7,8,9,11,14,15,17,19,21,22,23,24,26,27,29,6,10,12,18,28,3,13,16,20,25])*3 #20：5：5 -- first 20 for training, 5 for validation, last 5 for testing
reData = np.repeat(mRe, nframe-1)  #Repeat each Re number (nframe-1) times.
reData = (reData- np.min(reData))/(np.max(reData)-np.min(reData))  #Normalization
       
##Split datasets
xTrain, xVal, xTest = xData[0:20*(nframe-1),:,:], \
                      xData[20*(nframe-1):25*(nframe-1):,:], \
                      xData[25*(nframe-1):30*(nframe-1):,:]
yTrain, yVal, yTest = yData[0:20*(nframe-1),:,:], \
                      yData[20*(nframe-1):25*(nframe-1),:,:], \
                      yData[25*(nframe-1):30*(nframe-1),:,:]
reTrain, reVal, reTest = reData[0:20*(nframe-1)], \
                         reData[20*(nframe-1):25*(nframe-1)], \
                         reData[25*(nframe-1):30*(nframe-1)]


##Reshape to accommdate CNN inputs/output
xTrain = np.reshape(xTrain, (len(xTrain), nx, ny, 1)) 
xVal = np.reshape(xVal, (len(xVal), nx, ny, 1))   
xTest = np.reshape(xTest, (len(xTest), nx, ny, 1))  
yTrain = np.reshape(yTrain, (len(yTrain), nx, ny, 1)) 
yVal = np.reshape(yVal, (len(yVal), nx, ny, 1))  
yTest = np.reshape(yTest, (len(yTest), nx, ny, 1))
reTrain = np.reshape(reTrain, (len(reTrain), 1))
reVal = np.reshape(reVal, (len(reVal), 1))  
reTest = np.reshape(reTest, (len(reTest), 1))

print(np.shape(xTrain))
print(np.shape(xVal))
print(np.shape(reTest))
##Train Multi-input CNN
MICNN = oldMICNN(nx,ny)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='mean_squared_error')
img_file = 'architecture_oldMICNN.png'
plot_model(MICNN, to_file=img_file, show_shapes=True) #Plot MICNN architecture

model_checkpoint = ModelCheckpoint('weights_oldMICNN.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)

MICNN.fit([xTrain,reTrain], yTrain,
                    epochs=100,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([xVal,reVal], yVal))





    


    

    
    




