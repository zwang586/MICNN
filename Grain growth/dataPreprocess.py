# -*- coding: utf-8 -*-

##Process raw dataset - etaData2D.npy
#Extract grain structure pairs with desired time distance, based on deltaT.

import numpy as np

n = 600
nframe = 125
nx = 128
ny = 128


##Load all grain structure data - etaData2d.npy
##Shape: (600,125,128,128)
##600 simulations in total, each containing 125 frames of microstructures of 128x128 dimensions
etaData2D = np.load("data\\etaData2D_600.npy")  

##Processing by extracting grain structure pairs with desired time distance, based on deltaT.
deltaT = [1,3,4,5,7,9,11,12,13,15,17,18,19,21,22,24,25,27,29,30,2,6,14,20,28,8,10,16,23,26] ##20：5：5 -- first 20 for training, 5 for validation, last 5 for testing
maxDeltaT = np.max(deltaT) 

xData = np.ndarray((n,nframe-maxDeltaT,nx,ny),dtype = np.float32) #(600,95,128,128)
yData = np.ndarray((n,nframe-maxDeltaT,nx,ny),dtype = np.float32) #(600,95,128,128)  

print(np.shape(xData),np.shape(yData))   
for i in range(0,n):
    print(i)
    i_group = int(i/20)  #20 simulations for each deltaT sampling point.
    for j in range(0,nframe-maxDeltaT):
        xData[i,j,:,:] = etaData2D[i,j,:,:]
        yData[i,j,:,:] = etaData2D[i,j+int(deltaT[i_group]),:,:] #Postprocess based on delta_t

xData = np.reshape(xData,(n*(nframe-maxDeltaT),nx,ny))
yData = np.reshape(yData,(n*(nframe-maxDeltaT),nx,ny))

np.save("xData.npy", xData)
np.save("yData.npy", yData)