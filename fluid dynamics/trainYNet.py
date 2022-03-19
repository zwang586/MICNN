# -*- coding: utf-8 -*-
#Training and testing of the proposed gating-based yNet

import numpy as np
import matplotlib.pyplot as plt
from model import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

load_weights = False

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


##Train Multi-input CNN (MICNN)
MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='mean_squared_error')
img_file = 'architecture_yNet.png'
plot_model(MICNN, to_file=img_file, show_shapes=True) #Plot MICNN architecture

model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
if load_weights == True:
    MICNN.load_weights("weights_yNet.h5")
    print("Weights loaded!")
else:
    MICNN.fit([xTrain,reTrain], yTrain,
                        epochs=100,
                        batch_size=2,
                        shuffle=True,
                        callbacks=[model_checkpoint],
                        validation_data=([xVal,reVal], yVal))

decoded_imgs = MICNN.predict([xTest,reTest])


########Testing of yNet under 5 unused Re conditions#############
vMin = 0.2
vMax = 0.8
for i in range(0,5):
    
##Save five random testing results. 
#    test_frame_no = np.random.rand()*(nframe-1)
#    t_no = int(test_frame_no)+(nframe-1)*i
#    print(reTest[t_no,0]*(162-75)+75)    
    t_no = 118+159*i
    print(reTest[t_no,0]*(162-75)+75)
# Display input flow field
    ax = plt.imshow(np.rot90(xTest[t_no].reshape(nx, ny)),cmap = 'seismic', vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    
# Display evolved field: ground truth
    ax = plt.imshow(np.rot90(yTest[t_no].reshape(nx, ny)),cmap = 'seismic',vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
# display evolved field: MICNN prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx, ny)),cmap = 'seismic',vmin = vMin, vmax = vMax)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    
# Display error
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yTest[t_no].reshape(nx, ny)
    AError = (yPre*(valMax-valMin)+valMin) - (yGT*(valMax-valMin)+valMin)  
    ax = plt.imshow(np.rot90(AError[:,:]),cmap = 'seismic', vmin = -3, vmax = 3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testCylno'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()

####Calculate MSE and RMSE for all 5x159 testing cases
MSE_norm = 0.0
MSE = 0.0
RMSE = 0.0
for i in range(0,5*(nframe-1)):
    yP = decoded_imgs[i].reshape(nx, ny)
    yG = yTest[i].reshape(nx, ny)
    MSE_norm = np.mean((yPre- yGT)**2) + MSE_norm
    MSE = np.mean((yPre*(valMax-valMin) - yGT*(valMax-valMin))**2) + MSE
MSE_norm = MSE_norm/(5*(nframe-1))
MSE = MSE/(5*(nframe-1))
RMSE = np.sqrt(MSE)
print('MSE_norm',MSE_norm, 'MSE:', MSE,'rmse:',RMSE)



    


    

    
    


