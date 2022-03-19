# -*- coding: utf-8 -*-

##############test the trained yNet on other types of holes#######################
import numpy as np
import matplotlib.pyplot as plt
from model import *
from matplotlib import cm 


maxF = 50E6
minF = 20E6
nx = 128
ny = 128


xData = np.load('struct2D_extrap.npy')
yData = np.load('vmStress2D_extrap.npy')
yData[yData >= 1200E6] = 1200E6 ##Threshold using 1200E6
maxStress = 1200E6
minStress = 0
yData = (yData - minStress)/(maxStress-minStress)

forceData = np.ones(10,)
forceData = np.loadtxt('data_shapeExtrapolation\\F_extrap.txt')
  
forceData = (-forceData - minF)/(maxF-minF)        
    

plt.close()
  

xData = np.reshape(xData, (len(xData), nx, ny, 1)) 
yData = np.reshape(yData, (len(yData), nx, ny, 1))  
forceData = np.reshape(forceData, (len(forceData), 1))


  
MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")

decoded_imgs = MICNN.predict([xData,forceData])

allRMSE = np.zeros(10,)
new_hot_r = cm.get_cmap('hot_r', 3)
for i in range(0,10):

    t_no = i
    print("t_no",t_no)
    print("Force:",minF+(maxF-minF)*forceData[t_no])
    #Display perforation structure
    ax = plt.imshow(np.rot90(xData[t_no].reshape(nx, ny)),cmap = new_hot_r, extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_extrap_no'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display ground truth
    ax = plt.imshow(np.rot90(yData[t_no].reshape(nx, ny)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_extrap_no'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    
    # display prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx, ny)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_extrap_no'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")


    #Display yNet prediction
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yData[t_no].reshape(nx, ny)
    yError = yPre - yGT
    ax = plt.imshow(np.rot90(yError[:,:]),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = -0.017, vmax = 0.017)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testStress_extrap_no'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    allRMSE[i] = np.sqrt(np.mean(((yPre - yGT)*(maxStress-minStress))**2))
    
np.savetxt('RMSE_extrap.out',RMSE)
 