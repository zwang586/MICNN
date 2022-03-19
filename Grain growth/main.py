# -*- coding: utf-8 -*-

##Train and test yNet for grain growth simulation.

import numpy as np
import matplotlib.pyplot as plt
from model import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint

load_weights = False

n1 = 600 #Number of total simulations.
n2 = 20 #Number of simulations at each deltaT.
n3 = 95 #Number of extracted input-output pairs from each simulation.

nx  = 128
ny = 128

##laad xData and yData 
xData = np.load('xData.npy')
yData = np.load('yData.npy')
print("Load data done!")  


##Create deltaT dataset (deltaTData) based on the data-splitting scheme during data preprocessing
deltaT = [1,3,4,5,7,9,11,12,13,15,17,18,19,21,22,24,25,27,29,30,2,6,14,20,28,8,10,16,23,26] ##20：5：5 -- first 20 for training, 5 for validation, last 5 for testing
deltaTData = np.zeros(n1*n3,)
deltaTData = np.repeat(deltaT,n2*n3)
deltaTData = deltaTData/np.max(deltaT)#normalization 

##Split datasets
xTrain, xVal, xTest = xData[0:20*n2*n3,:,:], \
                      xData[20*n2*n3:25*n2*n3,:,:], \
                      xData[25*n2*n3:30*n2*n3,:,:]
yTrain, yVal, yTest = yData[0:20*n2*n3,:,:], \
                      yData[20*n2*n3:25*n2*n3,:,:], \
                      yData[25*n2*n3:30*n2*n3,:,:]
deltaTTrain, deltaTVal, deltaTTest = deltaTData[0:20*n2*n3], \
                                     deltaTData[20*n2*n3:25*n2*n3], \
                                     deltaTData[25*n2*n3:30*n2*n3]


##Reshape to accommdate CNN inputs/output
xTrain = np.reshape(xTrain, (len(xTrain), nx, ny, 1))  
xVal = np.reshape(xVal, (len(xVal), nx, ny, 1))  
xTest = np.reshape(xTest, (len(xTest), nx, ny, 1))   
yTrain = np.reshape(yTrain, (len(yTrain), nx, ny, 1))  
yVal = np.reshape(yVal, (len(yVal), nx, ny, 1))
yTest = np.reshape(yTest, (len(yTest), nx, ny, 1))  
deltaTTrain = np.reshape(deltaTTrain, (len(deltaTTrain), 1))
deltaTVal = np.reshape(deltaTVal, (len(deltaTVal), 1))
deltaTTest = np.reshape(deltaTTest, (len(deltaTTest), 1))


##Train Multi-input CNN (MICNN) 
MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='binary_crossentropy')

if load_weights == True:
    MICNN.load_weights("weights_yNet.h5")
else: 
    model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
    MICNN.fit([xTrain,deltaTTrain], yTrain,
                    epochs=100,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([xVal,deltaTVal], yVal))

decoded_imgs = MICNN.predict([xTest,deltaTTest])

####################5 random testing reulsts####
for i in range(0,5):
    rand1 = np.random.rand()*5*n2*n3
    t_no = int(rand1)
    #Display intinal grain structure
    ax = plt.imshow(xTest[t_no].reshape(nx, ny),cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testno'+str(i+1)+'_1.jpg', dpi=150, bbox_inches = "tight")
    
    # display evolved grain structure (ground truth)
    ax = plt.imshow(yTest[t_no].reshape(nx, ny),cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testno'+str(i+1)+'_2.jpg', dpi=150, bbox_inches = "tight")
    
    # display eovolved structure by yNet prediction
    ax = plt.imshow(decoded_imgs[t_no].reshape(nx, ny),cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testno'+str(i+1)+'_3.jpg', dpi=150, bbox_inches = "tight")
    
    print(deltaTTest[t_no,0]*np.max(deltaT))
    
    #Display error
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yTest[t_no].reshape(nx, ny)
    yError = np.absolute(yPre - yGT) 
    ax = plt.imshow(yError[:,:],cmap = 'coolwarm', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testno'+str(i+1)+'_4.jpg', dpi=150, bbox_inches = "tight")
    plt.close()



##############Calculate RMSE for 100*95 cases##########################
allMSE = np.zeros(5*n2*n3,)
allRMSE = np.zeros(5*n2*n3,)
for i in range(0,5*n2*n3):
    t_no = i
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yTest[t_no].reshape(nx, ny)
    MSE = np.mean((yPre - yGT)**2)
    allMSE[i] = MSE
    allRMSE[i] = np.sqrt(MSE)

print('Average RMSE:',np.mean(allRMSE))



