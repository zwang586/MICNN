# -*- coding: utf-8 -*-
import numpy as np
from nltk.tokenize import word_tokenize 

##Read all 5x192 testing data.
n = 5
nframe = 192
mRe = [3,8,17,20,24]  #No. of 5 tesing conditions
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
valMin = -352.145311756207
valMax = 352.146269050102

vortData2D = (vortData2D- valMin)/(valMax-valMin)


xData = np.zeros((n*(nframe-1),257,65))    
yData = np.zeros((n*(nframe-1),257,65))  

##Extract neighboring frames as input-output data pairs
count = 0  
for i in range(0,n):
    for j in range(0,nframe-1):
        xData[count,:,:] = vortData2D[i,j,:,:]
        yData[count,:,:] = vortData2D[i,j+1,:,:]
        count = count + 1

np.save("xData192.npy",xData)
np.save("yData192.npy",yData)

