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



nx = 1600
ny = 1600
delta_t = [1,2,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,26,26,27,28,30,3,10,21,25,29]
delta_t = [element * 4 for element in delta_t]

MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5") 


##########delta_T = 1#############################################
eta = np.zeros((nx,ny),dtype = np.float32)
eta1 = np.zeros((nx,ny),dtype = np.float32) 
eta = np.load("data seeding\\eta_initial_1.npy") 
  
x_test_0 = eta[:,:]
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
deltat_test_0 = delta_t[0]/60.0  #[16] = 20; [24] = 30;
deltat_test_0 = np.reshape(deltat_test_0, (1, 1))
###Recurrent prediction
for istep in range(0,65):
    print(istep)
    #Display intinal shape
    ax = plt.imshow(x_test_0.reshape(nx, ny),cmap = 'coolwarm', vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_title('$\mathit{\Delta}t$'+' = 1',loc = 'right', fontsize = 10)
    ax.axes.set_title('$\mathit{t}$'+'_'+'$\mathit{step}$'+' = '+str(istep),loc = 'left', fontsize = 10)
    plt.savefig('large_dt1_'+'eta_'+str(istep)+'.jpg', dpi=200, bbox_inches = "tight")
    plt.close()
    eta2D = x_test_0.reshape(nx, ny)
    x_test_1 = MICNN.predict([x_test_0,deltat_test_0])
    x_test_0 = x_test_1
 
              
##########delta_T = 5#############################################
eta = np.zeros((nx,ny),dtype = np.float32)
eta1 = np.zeros((nx,ny),dtype = np.float32) 
eta = np.load("data seeding\\eta_initial_5.npy") 
nx2 = 128
ny2 = 128   
x_test_0 = eta[:,:]
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
deltat_test_0 = delta_t[3]/60.0  #[16] = 20; [24] = 30;
deltat_test_0 = np.reshape(deltat_test_0, (1, 1))
###Recurrent prediction
for istep in range(0,65):
    print(istep)
    #Display intinal shape
    ax = plt.imshow(x_test_0.reshape(nx, ny),cmap = 'coolwarm', vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_title('$\mathit{\Delta}t$'+' = 5',loc = 'right', fontsize = 10)
    ax.axes.set_title('$\mathit{t}$'+'_'+'$\mathit{step}$'+' = '+str(istep),loc = 'left', fontsize = 10)
    plt.savefig('large_dt5_'+'eta_'+str(istep)+'.jpg', dpi=200, bbox_inches = "tight")
    plt.close()
    eta2D = x_test_0.reshape(nx, ny)
    x_test_1 = MICNN.predict([x_test_0,deltat_test_0])
    x_test_0 = x_test_1
 
              
##########delta_T = 30#############################################
eta = np.zeros((nx,ny),dtype = np.float32)
eta1 = np.zeros((nx,ny),dtype = np.float32) 
eta = np.load("data seeding\\eta_initial_30.npy") 
nx2 = 128
ny2 = 128   
x_test_0 = eta[:,:]
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
deltat_test_0 = delta_t[24]/60.0  #[16] = 20; [24] = 30;
deltat_test_0 = np.reshape(deltat_test_0, (1, 1))
###Recurrent prediction
for istep in range(0,65):
    print(istep)
    #Display intinal shape
    ax = plt.imshow(x_test_0.reshape(nx, ny),cmap = 'coolwarm', vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_title('$\mathit{\Delta}t$'+' = 30',loc = 'right', fontsize = 10)
    ax.axes.set_title('$\mathit{t}$'+'_'+'$\mathit{step}$'+' = '+str(istep),loc = 'left', fontsize = 10)
    plt.savefig('large_dt30_'+'eta_'+str(istep)+'.jpg', dpi=200, bbox_inches = "tight")
    plt.close()
    eta2D = x_test_0.reshape(nx, ny)
    x_test_1 = MICNN.predict([x_test_0,deltat_test_0])
    x_test_0 = x_test_1