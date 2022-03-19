# -*- coding: utf-8 -*-

##Use the trained yNet to perform large-scale grain growth simulation

import numpy as np
import matplotlib.pyplot as plt
from model import *


nx = 1600
ny = 1600
deltaT = [1,3,4,5,7,9,11,12,13,15,17,18,19,21,22,24,25,27,29,30,2,6,14,20,28,8,10,16,23,26]
deltaT = deltaT/np.max(deltaT)


MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5") 


##########delta_T = 1#############################################
eta = np.zeros((nx,ny),dtype = np.float32)
eta = np.load("data_seeding_1600x1600\\eta_initial_1.npy") 
  
x_test_0 = eta[:,:]
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
deltaT_test_0 = deltaT[0]  #[0] = 1, [3] = 5; [19] = 30;
deltaT_test_0 = np.reshape(deltaT_test_0, (1, 1))
###Recurrent prediction
for istep in range(0,65):
    print(istep)
    ax = plt.imshow(x_test_0.reshape(nx, ny),cmap = 'coolwarm', vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_title('$\mathit{\Delta}t$'+' = 1',loc = 'right', fontsize = 10)
    ax.axes.set_title('$\mathit{t}$'+'_'+'$\mathit{step}$'+' = '+str(istep),loc = 'left', fontsize = 10)
    plt.savefig('large_dt1_'+'eta_'+str(istep)+'.jpg', dpi=200, bbox_inches = "tight")
    plt.close()
    eta2D = x_test_0.reshape(nx, ny)
    x_test_1 = MICNN.predict([x_test_0,deltaT_test_0])
    x_test_0 = x_test_1
 
              
##########delta_T = 5#############################################
eta = np.zeros((nx,ny),dtype = np.float32)
eta = np.load("data_seeding_1600x1600\\eta_initial_5.npy") 
  
x_test_0 = eta[:,:]
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
deltaT_test_0 = deltaT[3]  #[0] = 1, [3] = 5; [19] = 30;
deltaT_test_0 = np.reshape(deltaT_test_0, (1, 1))
###Recurrent prediction
for istep in range(0,65):
    print(istep)
    ax = plt.imshow(x_test_0.reshape(nx, ny),cmap = 'coolwarm', vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_title('$\mathit{\Delta}t$'+' = 5',loc = 'right', fontsize = 10)
    ax.axes.set_title('$\mathit{t}$'+'_'+'$\mathit{step}$'+' = '+str(istep),loc = 'left', fontsize = 10)
    plt.savefig('large_dt5_'+'eta_'+str(istep)+'.jpg', dpi=200, bbox_inches = "tight")
    plt.close()
    eta2D = x_test_0.reshape(nx, ny)
    x_test_1 = MICNN.predict([x_test_0,deltaT_test_0])
    x_test_0 = x_test_1
 
              
##########delta_T = 30#############################################
eta = np.zeros((nx,ny),dtype = np.float32)
eta = np.load("data_seeding_1600x1600\\eta_initial_30.npy") 
nx2 = 128
ny2 = 128   
x_test_0 = eta[:,:]
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
deltaT_test_0 = deltaT[19]  #[0] = 1, [3] = 5; [19] = 30;
deltaT_test_0 = np.reshape(deltaT_test_0, (1, 1))
###Recurrent prediction
for istep in range(0,65):
    print(istep)
    ax = plt.imshow(x_test_0.reshape(nx, ny),cmap = 'coolwarm', vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_title('$\mathit{\Delta}t$'+' = 30',loc = 'right', fontsize = 10)
    ax.axes.set_title('$\mathit{t}$'+'_'+'$\mathit{step}$'+' = '+str(istep),loc = 'left', fontsize = 10)
    plt.savefig('large_dt30_'+'eta_'+str(istep)+'.jpg', dpi=200, bbox_inches = "tight")
    plt.close()
    eta2D = x_test_0.reshape(nx, ny)
    x_test_1 = MICNN.predict([x_test_0,deltaT_test_0])
    x_test_0 = x_test_1