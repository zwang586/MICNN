# -*- coding: utf-8 -*-

##Test the trained yNet in modeling small patches and long tracks under 5 different laser power conditions.
import numpy as np
import matplotlib.pyplot as plt
from model import *
from calGAcc import *

########################################Single-layer small patches###########################################
nx = 128  
ny = 150
nx2 = 128
ny2 = 128
eta1GT = np.zeros((5,nx,ny))

##process 5 physical simulation results of sintered structure  (Ground truth)
for i in range(0,5):
    filename = 'data_singleLayerTest\\patch\\case_'+str(int(1+i))+'_eta_final_1.txt'
    with open(filename) as f:
        lines = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta1GT[i,k,j] = lines[count]
            count = count + 1


eta0 = np.zeros((5,nx,ny))
##read 5 intial structures used in physical simulation
for i in range(0,5):
    filename = 'data_singleLayerTest\\patch\\case_'+str(int(1+i))+'_eta_1.txt'
    with open(filename) as f:
        lines = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta0[i,k,j] = lines[count]
            count = count + 1
            
##yNet-based prediction of sintered structure  
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")

eta1Pre = np.zeros((5,nx,ny))
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
    x_test[:,:] = eta0[nn,:,0:128]
    x_test = np.reshape(x_test, (1, nx2, ny2, 1))
        
    decoded_imgs = MICNN.predict([x_test,v_test,p_test])
    
    y_test = decoded_imgs[0].reshape(nx2, ny2)
    eta1Pre[nn,:,0:128] = y_test[:,:]            
          
#####Calculate pixel-based global accuracy for 5 laser conditions
allGAcc = np.zeros(5,)
for nn in range(0,5):
    s_eta_true = np.zeros((1,nx2,ny2))
    s_eta_pre = np.zeros((1,nx2,ny2))
    s_eta_true[0,:,:] = eta1GT[nn,:,0:128]
    s_eta_pre[0,:,:] = eta1Pre[nn,:,0:128]
    n_sample = 1
    allGAcc[nn] = calGAcc(s_eta_true,s_eta_pre)
    
print(np.mean(allGAcc))   
np.savetxt('GAcc_patch.out',allGAcc)




############################Single-layer long track###########################################################
nx = 640  
ny = 150
nx2 = 640
ny2 = 128
eta1GT = np.zeros((5,nx,ny))

##process 5 physical simulation results of sintered structure  (Ground truth)
for i in range(0,5):
    filename = 'data_singleLayerTest\\longTrack\\case_'+str(int(1+i))+'_eta_final_1.txt'
    with open(filename) as f:
        lines = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta1GT[i,k,j] = lines[count]
            count = count + 1


eta0 = np.zeros((5,nx,ny))
##read 5 intial structures used in physical simulation
for i in range(0,5):
    filename = 'data_singleLayerTest\\longTrack\\case_'+str(int(1+i))+'_eta_1.txt'
    with open(filename) as f:
        lines = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta0[i,k,j] = lines[count]
            count = count + 1
            
##yNet-based prediction of sintered structure  
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")

eta1Pre = np.zeros((5,nx,ny))
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
    x_test[:,:] = eta0[nn,:,0:128]
    x_test = np.reshape(x_test, (1, nx2, ny2, 1))
        
    decoded_imgs = MICNN.predict([x_test,v_test,p_test])
    
    y_test = decoded_imgs[0].reshape(nx2, ny2)
    eta1Pre[nn,:,0:128] = y_test[:,:]            
          
#####Calculate pixel-based global accuracy for 5 laser conditions
allGAcc = np.zeros(5,)
for nn in range(0,5):
    s_eta_true = np.zeros((1,nx2,ny2))
    s_eta_pre = np.zeros((1,nx2,ny2))
    s_eta_true[0,:,:] = eta1GT[nn,:,0:128]
    s_eta_pre[0,:,:] = eta1Pre[nn,:,0:128]
    n_sample = 1
    allGAcc[nn]  = calGAcc(s_eta_true,s_eta_pre)
    
np.savetxt('GAcc_longTrack.out',allGAcc)  
