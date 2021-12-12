# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:55:30 2021

@author: zwang
"""


#based on main02_stress.py, but test results under different shapes

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
from matplotlib import cm 


load_weights = True
xData = np.load('struct2D_extrap.npy')
yData = np.load('vmStress2D_extrap.npy')
print(xData.shape)
print(yData.shape)
print("Load data done!")
force = np.ones(10,)
filename = 'data extrapolation\\m_F_extrap.txt'
with open(filename) as f:
    forceData = f.readlines() 
    

for i in range(0,len(forceData)): 
    x = forceData[i]
    force[i] = x

        
force = (-force - 20000E3)/(50000E3-20000E3)        
    
nx = 128
ny = 128

x_data = xData
y_data = yData
y_data[y_data >= 1200E6] = 1200E6 ##Threshold using 1500E6
maxStress = np.max(y_data)
minStress = np.min(y_data)
y_data = (y_data - minStress)/(maxStress-minStress)

new_hot_r = cm.get_cmap('hot_r', 3)

plt.close()
  
nx2 = 128
ny2 = 128
x_data = np.reshape(x_data, (len(x_data), nx2, ny2, 1))  # adapt this if using `channels_first` image data format

y_data = np.reshape(y_data, (len(y_data), nx2, ny2, 1))  # adapt this if using `channels_first` image data format

force_test = np.reshape(force, (len(force), 1))


  
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")

decoded_imgs = MICNN.predict([x_data,force])


errorRMSE = np.zeros(10,)
for i in range(0,10):

    t_no = i
    print("t_no",t_no)
    print("Force:",20000E3+30000E3*force_test[t_no])
    #Display perforation structure
    ax = plt.imshow(np.rot90(x_data[t_no].reshape(nx2, ny2)),cmap = new_hot_r, extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_extrap_no'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display ground truth
    ax = plt.imshow(np.rot90(y_data[t_no].reshape(nx2, ny2)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_extrap_no'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    # display reconstruction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx2, ny2)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_extrap_no'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")


    #Display yNet prediction
    xP = decoded_imgs[t_no].reshape(nx2, ny2)
    xG = y_data[t_no].reshape(nx2, ny2)
    AError = xP - xG
    ax = plt.imshow(np.rot90(AError[:,:]),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = -0.017, vmax = 0.017)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testStress_extrap_no'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    errorRMSE[i] = np.sqrt(np.mean(((xP - xG)*(maxStress-minStress))**2))
    
np.savetxt('errorMSE_extrap.out',errorMSE)
 