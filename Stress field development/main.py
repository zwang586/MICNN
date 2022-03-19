# -*- coding: utf-8 -*-


#######train yNet for predicting stress-field########################
import numpy as np
import matplotlib.pyplot as plt
from model import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import cm 


load_weights = True
nTrain = 25 #Number of F sampling points for training.
nVal = 5 
nTest = 5
nTotal = nTrain + nVal + nTest
nSample = 200 #Number of structure samples at each F point.
maxF = 50E6
minF = 20E6
nx = 128
ny = 128


##Load structure and vmStress data
xData = np.load('struct2D.npy')
yData = np.load('vmStress2D.npy')

yData[yData >= 1200E6] = 1200E6 ##Threshold using 1200E6
maxStress = np.max(yData)
minStress = np.min(yData)
print('maxStress:',maxStress,'min',minStress)
yData = (yData - minStress)/(maxStress-minStress)


##Load force data
forceTrainVal = np.zeros((nTrain+nVal)*nSample,)
forceTest = np.zeros(nTest*nSample,)

forceTrainVal = np.loadtxt('data\\F_trainVal.txt')
forceTest = np.loadtxt('data\\F_test.txt')
    
force = np.concatenate((forceTrainVal,forceTest), axis = 0)  

##Normalization      
force = (-force - minF)/(maxF - minF) 
       
forceData = np.repeat(force, nSample)    

##Split datasets
xTrain, xVal, xTest = xData[0:nTrain*nSample,:,:], \
                      xData[nTrain*nSample:(nTrain+nVal)*nSample,:,:], \
                      xData[(nTrain+nVal)*nSample:nTotal*nSample,:,:]
yTrain, yVal, yTest = yData[0:nTrain*nSample,:,:], \
                      yData[nTrain*nSample:(nTrain+nVal)*nSample,:,:], \
                      yData[(nTrain+nVal)*nSample:nTotal*nSample,:,:]
forceTrain, forceVal, forceTest = forceData[0:nTrain*nSample], \
                                  forceData[nTrain*nSample:(nTrain+nVal)*nSample], \
                                  forceData[(nTrain+nVal)*nSample:nTotal*nSample]
print(np.max(forceTest))

##Reshape to accommdate CNN inputs/output
xTrain = np.reshape(xTrain, (len(xTrain), nx, ny, 1)) 
xVal = np.reshape(xVal, (len(xVal), nx, ny, 1)) 
xTest = np.reshape(xTest, (len(xTest), nx, ny, 1))  
yTrain = np.reshape(yTrain, (len(yTrain), nx, ny, 1))  
yVal = np.reshape(yVal, (len(yVal), nx, ny, 1))  
yTest = np.reshape(yTest, (len(yTest), nx, ny, 1)) 
forceTrain = np.reshape(forceTrain, (len(forceTrain), 1))
forceVal = np.reshape(forceVal, (len(forceVal), 1))
forceTest = np.reshape(forceTest, (len(forceTest), 1))


##Train multi-input CNN
MICNN = yNet(nx,ny)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='mean_squared_error')

if load_weights == True:
    MICNN.load_weights("weights_yNet.h5")
else: 
#    autoencoder.load_weights("test11v5.h5")
    model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
    autoencoder.fit([xTrain,forceTrain], yTrain,
                    epochs=100,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([xVal,forceVal], yVal))

decoded_imgs = MICNN.predict([xTest,forceTest])


############5 random testing results#################
new_hot_r = cm.get_cmap('hot_r', 3)
for i in range(0,5):
    rand1 = np.random.rand()*nTest*nSample
    t_no = int(rand1)
    print("t_no",t_no)
    print("Force:",minF+(maxF-minF)*forceTest[t_no])
    #Display perforation structure
    ax = plt.imshow(np.rot90(xTest[t_no].reshape(nx, ny)),cmap = new_hot_r, extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_no'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    
    # display ground truth by solid mechanics simulation
    ax = plt.imshow(np.rot90(yTest[t_no].reshape(nx, ny)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_no'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    # display yNet prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx, ny)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_no'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")


    #Display error
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yTest[t_no].reshape(nx, ny)
    yError = yPre - yGT
    ax = plt.imshow(np.rot90(yError[:,:]),cmap = 'bwr', extent=[0, 128, 0, 128], aspect = "equal", vmin = -0.0417, vmax = 0.0417)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testStress_no'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()



##Calculate RMSE for 5*200 case
allMSE = np.zeros(nTest*nSample,)
allRMSE = np.zeros(nTest*nSample,)
for i in range(0,nTest*nSample):
    t_no = i
    yPre = decoded_imgs[t_no].reshape(nx, ny)
    yGT = yTest[t_no].reshape(nx, ny)
    errorMSE = np.mean(((yPre - yGT)*(maxStress-minStress))**2)
    allMSE[i] = errorMSE
#    print(mMSE[i])
    allRMSE[i] = np.sqrt(errorMSE)

print("RMSE for all 1000 testing results:",np.mean(allRMSE))





