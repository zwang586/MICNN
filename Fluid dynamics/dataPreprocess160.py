# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Tue Mar.15 2022

@author: zwang
"""
##Read all 30x160 data, including both training and tesing data.
n = 30
nframe = 160
mRe = 75+np.array([0,1,2,4,5,7,8,9,11,14,15,17,19,21,22,23,24,26,27,29,6,10,12,18,28,3,13,16,20,25])*3 ##20：5：5 -- first 20 for training, 5 for validation, last 5 for testing
vortData = np.zeros((n,nframe,257,65))
for i in range(0,n):
    Re = mRe[i]
    print('Preprocessing Re=',Re)
    for j in range(0,nframe):
        filename = 'data\\'+str(Re)+'\\'+str(j)+'.txt' #Raw txt file: 65 lines 
    
        with open(filename) as f:
            lines = f.readlines()

        for k in range(0,len(lines)):
            line = lines[k]
            lineSplit = line.split(',')
            vortData[i,j,:,k] = lineSplit


###Clean data            
##Remove cylinder data
for i in range(0,n):
    for j in range(0,nframe):
        for k in range(0,257):
            for l in range(0,65):
                if (k-43)**2+(l-32.5)**2<=17**2: #Vorticity inside cylinder == 0
                    vortData[i,j,k,l] = 0.0
vortData2D = vortData


##min-max normalization
valMin = np.min(vortData)
valMax = np.max(vortData)

print(valMin,valMax)
vortData2D = (vortData2D- valMin)/(valMax-valMin)
print(np.min(vortData2D),np.max(vortData2D))


##Extract neighboring frames as input-output data pairs
xData = np.zeros((n*(nframe-1),257,65))    
yData = np.zeros((n*(nframe-1),257,65))  
count = 0  
for i in range(0,n):
    for j in range(0,nframe-1):
        xData[count,:,:] = vortData2D[i,j,:,:]
        yData[count,:,:] = vortData2D[i,j+1,:,:]
        count = count + 1

np.save("xData.npy",xData)
np.save("yData.npy",yData)

