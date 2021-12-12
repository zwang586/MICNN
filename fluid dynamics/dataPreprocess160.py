# -*- coding: utf-8 -*-
import numpy as np
from nltk.tokenize import word_tokenize 

##Read all 29x160 data, including both training and tesing data.
n = 29
nframe = 160
mRe = [0,1,2,4,5,6,7,9,10,11,12,13,14,15,16,18,19,21,22,23,25,26,27,28,3,8,17,20,24]  #first 24 for training, last 5 for testing
vortData = np.zeros((n,nframe,257,65))
for i in range(0,n):
    print(i)
    Re = int(75+ mRe[i]*3)
    print('Re = ',Re)
    for j in range(0,nframe):
        filename = 'data\\'+str(Re)+'\\'+str(j)+'.txt'
    
        with open(filename) as f:
            x = f.readlines()

        for k in range(0,len(x)):
            xx = x[k]
            xxx = xx.split(',')
#            print(xxx)
            vortData[i,j,:,k] = xxx
            
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

xData = np.zeros((n*(nframe-1),257,65))    
yData = np.zeros((n*(nframe-1),257,65))  

##Extract neighboring frames as input-output data pairs
count = 0  
for i in range(0,n):
    for j in range(0,nframe-1):
        xData[count,:,:] = vortData2D[i,j,:,:]
        yData[count,:,:] = vortData2D[i,j+1,:,:]
        count = count + 1

np.save("xData.npy",xData)
np.save("yData.npy",yData)

