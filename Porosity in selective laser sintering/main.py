# -*- coding: utf-8 -*-
##Train and test yNet for porosity prediction

import numpy as np
import matplotlib.pyplot as plt
from model import *
from calGAcc import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint


load_weight = False
nTrain = 2250
nVal = 750
nTest = 750
nTotal = nTrain+nVal+nTest
nPatch = 58 #Number of patches that can be extracted from a 700-pixel long sintering simulation.
nx = 128
ny = 128
pMax = 40 #Unit: W
pMin = 20 
vMax = 2.5 #Unit:m s-1
vMin = 0.5

##Load raw structure data
##etaTrainVal:3000 long tracks before sintering. Shape:(3000,700,175), where first 2250 for training, the remaining 750 for validation.
##eta1TrainVal: the corresponding 3000 long tracks after sintering under 100 different laser conditions. Shape:(3000,700,175)
etaTrainVal = np.load('data\\etaTrainVal.npy')  
eta1TrainVal = np.load('data\\eta1TrainVal.npy')

##etaTest:750 long tracks before sintering. Shape:(750,700,175).
##eta1TrainVal: the corresponding 750 long tracks after sintering under 25 different laser conditions. Shape:(3000,700,175)
etaTest = np.load('data\\etaTest.npy')  
eta1Test = np.load('data\\eta1Test.npy')

eta = np.concatenate((etaTrainVal,etaTest), axis = 0)   #Shape: (3750,700,175). 
eta1 = np.concatenate((eta1TrainVal,eta1Test), axis = 0)  #Shape: (3750,700,175). 


##Cropping every 10 pixels 
##Generating 58 standard square structure pairs before and after sintering for each long track.
no = 0
xData = np.ndarray((nPatch*nTotal,nx,ny),dtype = np.float32) 
yData = np.ndarray((nPatch*nTotal,nx,ny),dtype = np.float32) 
for n in range(0,nTotal):  #loop over structures
        for m in range(0,nPatch): #loop along x direction in each structure
            xData[no,:,:] = eta[n,(10*m):(10*m+ny),22:150] 
            yData[no,:,:] = eta1[n,(10*m):(10*m+ny),22:150]
            no = no+1


##Load applied laser condition data in .txt
pTrainTxt = np.loadtxt('data\\pTrain.txt')
vTrainTxt = np.loadtxt('data\\vTrain.txt') 
pValTxt = np.loadtxt('data\\pVal.txt')
vValTxt = np.loadtxt('data\\vVal.txt') 
pTestTxt = np.loadtxt('data\\pTest.txt')
vTestTxt = np.loadtxt('data\\vTest.txt')

pTxt = np.concatenate((pTrainTxt,pValTxt,pTestTxt), axis = 0)
vTxt = np.concatenate((vTrainTxt,vValTxt,vTestTxt), axis = 0)  

##Normalization
pTxt = (pTxt - pMin)/(pMax - pMin)
vTxt = (vTxt - vMin)/(vMax - vMin) 
   
##Create p and v datasets
pData = np.repeat(pTxt, nPatch)
vData = np.repeat(vTxt, nPatch) 


##Split datasets
xTrain, xVal, xTest = xData[0:2250*nPatch,:,:], \
                      xData[2250*nPatch:3000*nPatch,:,:], \
                      xData[3000*nPatch:3750*nPatch,:,:]
yTrain, yVal, yTest = yData[0:2250*nPatch,:,:], \
                      yData[2250*nPatch:3000*nPatch,:,:], \
                      yData[3000*nPatch:3750*nPatch,:,:]
pTrain, pVal, pTest = pData[0:2250*nPatch], \
                      pData[2250*nPatch:3000*nPatch], \
                      pData[3000*nPatch:3750*nPatch]
vTrain, vVal, vTest = vData[0:2250*nPatch], \
                      vData[2250*nPatch:3000*nPatch], \
                      vData[3000*nPatch:3750*nPatch]
                                     
print(np.shape(xTrain),np.shape(xVal),np.shape(xTest))                                     
##Reshape to accommdate CNN inputs/output
xTrain = np.reshape(xTrain, (len(xTrain), nx, ny, 1)) 
xVal = np.reshape(xVal, (len(xVal), nx, ny, 1)) 
xTest = np.reshape(xTest, (len(xTest), nx, ny, 1))  
yTrain = np.reshape(yTrain, (len(yTrain), nx, ny, 1)) 
yVal = np.reshape(yVal, (len(yVal), nx, ny, 1)) 
yTest = np.reshape(yTest, (len(yTest), nx, ny, 1))  
yTrain = np.reshape(yTrain, (len(yTrain), nx, ny, 1)) 
yVal = np.reshape(yVal, (len(yVal), nx, ny, 1)) 
yTest = np.reshape(yTest, (len(yTest), nx, ny, 1)) 
pTrain = np.reshape(pTrain, (len(pTrain), 1))
pVal = np.reshape(pVal, (len(pVal), 1))
pTest = np.reshape(pTest, (len(pTest), 1))
vTrain = np.reshape(vTrain, (len(vTrain), 1))
vVal = np.reshape(vVal, (len(vVal), 1))
vTest = np.reshape(vTest, (len(vTest), 1))

#############Training
MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.compile(optimizer='adadelta', loss='binary_crossentropy')
model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
if load_weight == True:
    MICNN.load_weights("weights_yNet.h5")
    print('Weights loaded!')
else:   
    MICNN.fit([xTrain,vTrain,pTrain], yTrain,
                    epochs=100,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([xVal,vVal,pVal], yVal))


decoded_imgs = MICNN.predict([xTest,vTest,pTest])


for i in range(0,5):
    print(i+1)
    rand1 = np.random.rand()*nPatch*nTest
    t_no = int(rand1)
    #Display intinal structure
    ax = plt.imshow(np.rot90(xTest[t_no].reshape(nx, ny)),cmap = 'gray', vmin = 0.0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display GT
    ax = plt.imshow(np.rot90(yTest[t_no].reshape(nx, ny)),cmap = 'gray', vmin = 0.0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display yNet prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx, ny)),cmap = 'gray', vmin = 0.0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    #Display error:
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yTest[t_no].reshape(nx, ny)
    yError = np.absolute(yPre - yGT) 
    yError[yError >= 0.5] = 1.0
    yError[yError < 0.5] = 0.0
    ax = plt.imshow(np.rot90(yError[:,:]),cmap = 'gray', vmin = 0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    print('P:',pTest[t_no,0]*20+20)
    print('V:',vTest[t_no,0]*2.0+0.5)
 
    
##Calculate global accuracy for all testing results.
s_eta_true = np.reshape(yTest, (len(yTest), nx, ny))
s_eta_pre = np.reshape(decoded_imgs, (len(decoded_imgs), nx, ny))
meanGAcc = calGAcc(s_eta_true,s_eta_pre)
print(meanGAcc)

