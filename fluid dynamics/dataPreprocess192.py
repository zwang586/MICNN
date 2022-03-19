# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Tue Mar.15 2022

@author: zwang
"""
##Read all 5x192 testing data.
n = 5
nframe = 192
mRe = 75+np.array([3,13,16,20,25])*3  #No. of 5 tesing conditions
vortData = np.zeros((n,nframe,257,65))
for i in range(0,n):
    Re = mRe[i]
    print('Preprocessing Re=',Re)
    for j in range(0,nframe):
        filename = 'data\\'+str(Re)+'\\'+str(j)+'.txt'
    
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
valMin = -355.570965825434
valMax = 355.570406560352
vortData2D = (vortData2D- valMin)/(valMax-valMin)

 
##Extract neighboring frames as input-output data pairs
xData = np.zeros((n*(nframe-1),257,65))    
yData = np.zeros((n*(nframe-1),257,65))
count = 0  
for i in range(0,n):
    for j in range(0,nframe-1):
        xData[count,:,:] = vortData2D[i,j,:,:]
        yData[count,:,:] = vortData2D[i,j+1,:,:]
        count = count + 1

np.save("xData192.npy",xData)
np.save("yData192.npy",yData)

