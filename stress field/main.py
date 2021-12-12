# -*- coding: utf-8 -*-


#######train yNet for predicting stress-field

import os

import numpy as np
import matplotlib.pyplot as plt
from model import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import cm 


load_weights = True
xData = np.load('struct2D.npy')
yData = np.load('vmStress2D.npy')
print(xData.shape)
print(yData.shape)
print("Load data done!")
force = np.ones(30*200,)
filename = 'data\\m_F.txt' ###Read applied force sampling data
with open(filename) as f:
    forceData = f.readlines() 
    
no = 0
for i in range(0,len(forceData)): 
    x = forceData[i]
    for j in range(0,200): 
        force[no] = x
        no = no + 1
        
force = (-force - 20000E3)/(50000E3-20000E3)        
    
nx = 128
ny = 128

x_data = xData
y_data = yData
y_data[y_data >= 1200E6] = 1200E6 ##Threshold using 1500E6
maxStress = np.max(y_data)
minStress = np.min(y_data)
y_data = (y_data - minStress)/(maxStress-minStress)

x_train = x_data[0*200:25*200,:,:]
x_test = x_data[25*200:30*200,:,:]
y_train = y_data[0*200:25*200,:,:]
y_test = y_data[25*200:30*200,:,:]
force_train = force[0*200:25*200]
force_test = force[25*200:30*200]
 
print(x_train.shape)
print(x_test.shape)
print(force_train.shape)

new_hot_r = cm.get_cmap('hot_r', 3)
 
nx2 = 128
ny2 = 128
x_train = np.reshape(x_train, (len(x_train), nx2, ny2, 1)) 
x_test = np.reshape(x_test, (len(x_test), nx2, ny2, 1))  
y_train = np.reshape(y_train, (len(y_train), nx2, ny2, 1))  
y_test = np.reshape(y_test, (len(y_test), nx2, ny2, 1)) 
force_train = np.reshape(force_train, (len(force_train), 1))
force_test = np.reshape(force_test, (len(force_test), 1))


  
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.compile(optimizer='Adam', loss='mean_squared_error')
dot_img_file = 'architecture_yNet.png'
tf.keras.utils.plot_model(MICNN, to_file=dot_img_file, show_shapes=True)
if load_weights == True:
    MICNN.load_weights("weights_yNet.h5")
else: 
#    autoencoder.load_weights("test11v5.h5")
    model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
    autoencoder.fit([x_train,force_train], y_train,
                    epochs=100,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([x_test,force_test], y_test))

decoded_imgs = MICNN.predict([x_test,force_test])


############5 random testing results#################
for i in range(0,5):
    rand1 = np.random.rand()*5*200
    t_no = int(rand1)
    print("t_no",t_no)
    print("Force:",20000E3+30000E3*force_test[t_no])
    #Display perforation structure
    ax = plt.imshow(np.rot90(x_test[t_no].reshape(nx2, ny2)),cmap = new_hot_r, extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_no'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    
    # display ground truth by solid mechanics simulation
    ax = plt.imshow(np.rot90(y_test[t_no].reshape(nx2, ny2)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_no'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    # display yNet prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx2, ny2)),cmap = 'jet', extent=[0, 128, 0, 128], aspect = "equal", vmin = 0, vmax = 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('testStress_no'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")


    #Display error
    xP = decoded_imgs[t_no].reshape(nx2, ny2)
    xG = y_test[t_no].reshape(nx2, ny2)
    AError = xP - xG
    ax = plt.imshow(np.rot90(AError[:,:]),cmap = 'bwr', extent=[0, 128, 0, 128], aspect = "equal", vmin = -0.0417, vmax = 0.0417)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
#    plt.colorbar()
    plt.savefig('testStress_no'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()



##Calculate RMSE for 5*200 case
mMSE = np.zeros(5*200,)
mRMSE = np.zeros(5*200,)
for i in range(0,5*200):
    x_test_0_2D_pre = np.zeros((nx2,ny2))
    x_test_0_2D_GT = np.zeros((nx2,ny2))
    t_no = i
    x_test_0_2D_pre = decoded_imgs[t_no].reshape(nx2, ny2)
    x_test_0_2D_GT = y_test[t_no].reshape(nx2, ny2)
    errorMSE = np.mean(((x_test_0_2D_pre - x_test_0_2D_GT)*(maxStress-minStress))**2)
    mMSE[i] = errorMSE
#    print(mMSE[i])
    mRMSE[i] = np.sqrt(errorMSE)

print("RMSE for all 1000 testing results:",np.mean(mRMSE))





