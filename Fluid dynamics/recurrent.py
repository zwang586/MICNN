# -*- coding: utf-8 -*-

##Calculate predictive error over long-term simulation;


import numpy as np
import matplotlib.pyplot as plt
from model import *
import tensorflow as tf


n = 30  #number of Re sampling points
nframe = 160
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
                      xData[20*(nframe-1):25*(nframe-1),:,:], \
                      xData[25*(nframe-1):30*(nframe-1),:,:]
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

MICNN = yNet(nx,ny)
MICNN.load_weights("weights_yNet.h5")


###########Recurrent prediction by yNet#######################################
testNo = 1 ##testNo = 1,2,3,4,5

vMin = 0.2
vMax = 0.8
x_test_0 = xData[127+159*(24+testNo),:,:]   #Staring from frame No.128
x_test_0 = np.reshape(x_test_0, (1, nx, ny, 1))
rey_test_0 = reTest[127+159*(testNo-1),:]
print("rey:",75+(162-75)*rey_test_0)
rey_test_0 = np.reshape(rey_test_0, (1, 1))
allRMSE = np.zeros(65,)

yData192 = np.load('yData192.npy')
yData192 = yData192[:,0:256,0:64]

for i in range(0,65):
    print(i)

    ##Display and save yNet prediction
    ax = plt.imshow(np.rot90(x_test_0.reshape(nx, ny)),cmap = 'seismic', vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('gif_pre_testNo_'+str(testNo)+"_"+str(i+128)+'_2.jpg', dpi=300, bbox_inches = "tight")
    
    
    yPre = x_test_0.reshape(nx,ny)
    yGT = yData192[int(191*(testNo-1)+i+126),:,:]
    MSE = np.mean((yPre*(valMax-valMin) - yGT*(valMax-valMin))**2)
    RMSE = np.sqrt(MSE)
    print('RMSE:',RMSE)
    allRMSE[i] = RMSE
    
    x_test_1 = MICNN.predict([x_test_0,rey_test_0])
    x_test_0 = x_test_1
    

    
    ##Display and save physical simulation results
    ax = plt.imshow(np.rot90(yData192[int(191*(testNo-1)+i+126),:,:]),cmap = 'seismic', vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('gif_ori_testNo_'+str(testNo)+"_"+str(i+128)+'_2.jpg', dpi=300, bbox_inches = "tight")
    plt.close()

    
np.savetxt('recurrent_RMSE.out', allRMSE)

    

    
    


