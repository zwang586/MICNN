# -*- coding: utf-8 

###Compare the testing results of yNet and conventional MICNN at a random testing point

import numpy as np
import matplotlib.pyplot as plt
from model import *
from tensorflow.keras.models import *


n = 30  #number of Re sampling points
nframe = 160  #frames of each simulation
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


##Load the trained yNet and oldMICNN
MICNN1 = yNet(nx,ny)
MICNN1.load_weights("weights_yNet.h5")
decoded_imgs1 = MICNN1.predict([xTest,reTest])

MICNN2 = oldMICNN(nx,ny)
MICNN2.load_weights("weights_oldMICNN.h5")
decoded_imgs2 = MICNN2.predict([xTest,reTest])


########Testing of yNet and oldMICNN under a random testing point#################

t_no = int(np.random.rand()*5*(nframe-1))
vMin = 0.2
vMax = 0.8  
#    Display input flow field
ax = plt.imshow(np.rot90(xTest[t_no].reshape(nx, ny)),cmap = 'seismic', vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_1.jpg', dpi=300, bbox_inches = "tight")
plt.close()

#    Display evolved field: ground truth
ax = plt.imshow(np.rot90(yTest[t_no].reshape(nx, ny)),cmap = 'seismic',vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_2.jpg', dpi=300, bbox_inches = "tight")
plt.close()

#    Display evolved field: yNet prediction
ax = plt.imshow(np.rot90(decoded_imgs1[t_no].reshape(nx, ny)),cmap = 'seismic',vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_yNet_3.jpg', dpi=300, bbox_inches = "tight")
plt.close()

#   Display evolved field: yNet prediction
ax = plt.imshow(np.rot90(decoded_imgs2[t_no].reshape(nx, ny)),cmap = 'seismic',vmin = vMin, vmax = vMax)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('testCylno_oldMICNN_3.jpg', dpi=300, bbox_inches = "tight")
plt.close()




