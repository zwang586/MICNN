# -*- coding: utf-8 

###Compare the testing results of yNet and conventional MICNN at a random testing point

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

MICNN1 = yNet(nx2,ny2)
MICNN1.load_weights("weights_yNet.h5")
decoded_imgs1 = MICNN1.predict([x_test,rey_test])

MICNN2 = oldMICNN(nx2,ny2)
MICNN2.load_weights("weights_oldMICNN.h5")
decoded_imgs2 = MICNN2.predict([x_test,rey_test])


########Testing of yNet and oldMICNN under a random testing point#################

t_no = int(np.random.rand()*5*(nframe-1))
vMin = 0.3
vMax = 0.7  
#    Display input flow field
ax = plt.imshow(np.rot90(x_test[t_no].reshape(nx2, ny2)),cmap = 'seismic', vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_1.jpg', dpi=300, bbox_inches = "tight")
plt.close()

#    Display evolved field: ground truth
ax = plt.imshow(np.rot90(y_test[t_no].reshape(nx2, ny2)),cmap = 'seismic',vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_2.jpg', dpi=300, bbox_inches = "tight")
plt.close()

#    Display evolved field: yNet prediction
ax = plt.imshow(np.rot90(decoded_imgs1[t_no].reshape(nx2, ny2)),cmap = 'seismic',vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_yNet_3.jpg', dpi=300, bbox_inches = "tight")
plt.close()

#   Display evolved field: yNet prediction
ax = plt.imshow(np.rot90(decoded_imgs2[t_no].reshape(nx2, ny2)),cmap = 'seismic',vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_oldMICNN_3.jpg', dpi=300, bbox_inches = "tight")
plt.close()




