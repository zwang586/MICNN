# -*- coding: utf-8 -*-

import numpy as np
nx = 128
ny = 128
etaData2D = np.load("data\\etaData2D_600.npy")  #(600,125,128,128)
delta_t = [1,2,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,26,26,27,28,30,3,10,21,25,29]
print(delta_t)
xData = np.ndarray((600,95,nx,ny),dtype = np.float16)   
yData = np.ndarray((600,95,nx,ny),dtype = np.float16)     
for i in range(0,600):
    i_group = int(i/20)
    for j in range(0,95):
        xData[i,j,:,:] = etaData2D[i,j,:,:]
        yData[i,j,:,:] = etaData2D[i,j+int(delta_t[i_group]),:,:] #Postprocess based on delta_t

np.save("xData.npy",xData)
np.save("yData.npy",yData)