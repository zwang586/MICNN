# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from model import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from calG import *

########################################Single-layer small patches###########################################
nx = 128  #Adapt in all related places 
ny = 150
nx2 = 128
ny2 = 128
eta_final_GT = np.zeros((5,nx,ny))

#######process physical simulation results of sintered structure  (Ground truth)
for i in range(0,5):
    filename = 'data singleLayerVal\\patch\\case_'+str(int(1+i))+'_eta_final_1.txt'
    with open(filename) as f:
        x = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta_final_GT[i,k,j] = x[count]
            count = count + 1


eta_1_GT = np.zeros((5,nx,ny))
eta_final_pre = np.zeros((5,nx,ny))
#######read intial structure used in physical simulation
for i in range(0,5):
    filename = 'data singleLayerVal\\patch\\case_'+str(int(1+i))+'_eta_1.txt'
    with open(filename) as f:
        x = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta_1_GT[i,k,j] = x[count]
            count = count + 1

   
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")
########yNet-based prediction of sintered structure
for nn in range(0,5):
    eta = np.zeros((nx,ny))
    eta1 = np.zeros((nx,ny))
    p_test = 20+nn*5
    v_test = 0.5
    
    p_test = (p_test-20.0)/20.0
    v_test = (v_test-0.5)/2.0  
    
    v_test = np.reshape(v_test, (1, 1))
    p_test = np.reshape(p_test, (1, 1))
    
    x_test = np.zeros((nx2,ny2))
    x_test[:,:] = eta_1_GT[nn,:,0:128]
    x_test = np.reshape(x_test, (1, nx2, ny2, 1))
        
    decoded_imgs = MICNN.predict([x_test,v_test,p_test])
    
    y_test = decoded_imgs[0].reshape(nx2, ny2)
    eta_final_pre[nn,:,0:128] = y_test[:,:]            
          
#####Calculate pixel-based global accuracy
mPAcc = np.zeros(5,)
for nn in range(0,5):
    s_eta_true = np.zeros((1,nx2,ny2))
    s_eta_pre = np.zeros((1,nx2,ny2))
    s_eta_true[0,:,:] = eta_final_GT[nn,:,0:128]
    s_eta_pre[0,:,:] = eta_final_pre[nn,:,0:128]
    n_sample = 1
    PAcc = calG(s_eta_true,s_eta_pre,nx2,ny2,n_sample)
    mPAcc[nn] = PAcc
print(np.mean(mPAcc))   
np.savetxt('PAcc_patch.out',mPAcc)




############################Single-layer long track###########################################################
nx = 640  #Adapt in all related places 
ny = 150
nx2 = 640
ny2 = 128
eta_final_GT = np.zeros((5,nx,ny))

#######process physical simulation results of sintered structure  (Ground truth)
for i in range(0,5):
    filename = 'data singleLayerVal\\longTrack\\case_'+str(int(1+i))+'_eta_final_1.txt'
    with open(filename) as f:
        x = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta_final_GT[i,k,j] = x[count]
            count = count + 1


eta_1_GT = np.zeros((5,nx,ny))
eta_final_pre = np.zeros((5,nx,ny))
#######read intial structure used in physical simulation
for i in range(0,5):
    filename = 'data singleLayerVal\\longTrack\\case_'+str(int(1+i))+'_eta_1.txt'
    with open(filename) as f:
        x = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta_1_GT[i,k,j] = x[count]
            count = count + 1

   
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")
########yNet-based prediction of sintered structure
for nn in range(0,5):
    eta = np.zeros((nx,ny))
    eta1 = np.zeros((nx,ny))
    p_test = 20+nn*5
    v_test = 0.5
    
    p_test = (p_test-20.0)/20.0
    v_test = (v_test-0.5)/2.0  
    
    v_test = np.reshape(v_test, (1, 1))
    p_test = np.reshape(p_test, (1, 1))
    
    x_test = np.zeros((nx2,ny2))
    x_test[:,:] = eta_1_GT[nn,:,0:128]
    x_test = np.reshape(x_test, (1, nx2, ny2, 1))
        
    decoded_imgs = MICNN.predict([x_test,v_test,p_test])
    
    y_test = decoded_imgs[0].reshape(nx2, ny2)
    eta_final_pre[nn,:,0:128] = y_test[:,:]            
        
#####Calculate pixel-based global accuracy
mPAcc = np.zeros(5,)
for nn in range(0,5):
    s_eta_true = np.zeros((1,nx2,ny2))
    s_eta_pre = np.zeros((1,nx2,ny2))
    s_eta_true[0,:,:] = eta_final_GT[nn,:,0:128]
    s_eta_pre[0,:,:] = eta_final_pre[nn,:,0:128]
    n_sample = 1
    PAcc = calG(s_eta_true,s_eta_pre,nx2,ny2,n_sample)
    mPAcc[nn] = PAcc
print(np.mean(mPAcc))   
np.savetxt('PAcc_longTrack.out',mPAcc)