# -*- coding: utf-8 -*-
##Train and/or test yNet for porosity prediction

import os

import numpy as np
import matplotlib.pyplot as plt
from model import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split



load_weight = True
nTrain = 2250
nTest = 750
nTotal = 3000



########Data preprocess: cropping raw data to small standard patches####################3
eta = np.load('data\\eta.npy')  #
eta1 = np.load('data\\eta1.npy')

p = np.zeros(58*3000,)
v = np.zeros(58*3000,)

p_data = np.loadtxt('data\\pTrain.txt')
v_data = np.loadtxt('data\\vTrain.txt') 
no = 0
for n in range(0,2250):
    for m in range(0,58):
        p[no] = p_data[n]
        v[no] = v_data[n]
        no = no+1
          
p_data = np.loadtxt('data\\pTest.txt')
v_data = np.loadtxt('data\\vTest.txt')     
no = 0      
for n in range(0,750):
    for m in range(0,58):
        p[no+58*2250] = p_data[n]
        v[no+58*2250] = v_data[n]  
        no = no+1
##Normalization
p = (p-20.0)/20.0
v = (v-0.5)/2.0    
 

##Cropping##############
nx = 128
ny = 128
no = 0
x_data = np.ndarray((58*3000,nx,ny),dtype = np.float32) 
y_data = np.ndarray((58*3000,nx,ny),dtype = np.float32) 
for n in range(0,3000):  #loop over structures
        for m in range(0,58): #loop along x direction in each structure
            x_data[no,:,:] = eta[n,(10*m):(10*m+ny),22:150]  
            no = no+1
            
plt.imshow(x_data[1000,:,:], cmap='gray')
plt.show()
plt.imshow(x_data[2000,:,:], cmap='gray')
plt.show()

no = 0
for n in range(0,3000):  #loop over structures
        for m in range(0,58): #loop along x direction in each structure
            y_data[no,:,:] = eta1[n,(10*m):(10*m+ny),22:150]  
            no = no+1
            
plt.imshow(y_data[1000,:,:], cmap='gray')
plt.show()
plt.imshow(y_data[2000,:,:], cmap='gray')
plt.show()
 


x_train = x_data[0:58*nTrain,:,:]
x_train_label = y_data[0:58*nTrain,:,:]
x_test = x_data[58*2250:58*nTotal,:,:]
x_test_label = y_data[58*2250:58*nTotal,:,:]
p_train = p[0:58*nTrain]
p_test = p[58*2250:58*nTotal]
v_train = p[0:58*nTrain]
v_test = p[58*2250:58*nTotal]



nx2 = 128
ny2 = 128
x_train = np.reshape(x_train, (len(x_train), nx2, ny2, 1)) 
x_test = np.reshape(x_test, (len(x_test), nx2, ny2, 1))  
x_train_label = np.reshape(x_train_label, (len(x_train_label), nx2, ny2, 1)) 
x_test_label = np.reshape(x_test_label, (len(x_test_label), nx2, ny2, 1))  
v_train = np.reshape(v_train, (len(v_train), 1))
v_test = np.reshape(v_test, (len(v_test), 1))
p_train = np.reshape(p_train, (len(p_train), 1))
p_test = np.reshape(p_test, (len(p_test), 1))

#############Training
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.compile(optimizer='adadelta', loss='binary_crossentropy')
model_checkpoint = ModelCheckpoint('weights_yNet.h5',monitor='val_loss', save_weights_only=True, mode='min', save_best_only = True)
if load_weight == True:
    MICNN.load_weights("weights_yNet.h5")
    print('Weights loaded!')
else:   
    MICNN.fit([x_train,v_train,p_train], x_train_label,
                    epochs=15,
                    batch_size=2,
                    shuffle=True,
                    callbacks=[model_checkpoint],
                    validation_data=([x_test,v_test,p_test], x_test_label))


decoded_imgs = MICNN.predict([x_test,v_test,p_test])


for i in range(0,5):
    print(i+1)
    rand1 = np.random.rand()*58*750
    t_no = int(rand1)
    #Display intinal structure
    ax = plt.imshow(np.rot90(x_test[t_no].reshape(nx2, ny2)),cmap = 'gray', vmin = 0.0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_1.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display GT
    ax = plt.imshow(np.rot90(x_test_label[t_no].reshape(nx2, ny2)),cmap = 'gray', vmin = 0.0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_2.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    # display yNet prediction
    ax = plt.imshow(np.rot90(decoded_imgs[t_no].reshape(nx2, ny2)),cmap = 'gray', vmin = 0.0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_3.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    #Display error:
    xP = decoded_imgs[t_no].reshape(nx2, ny2)
    xG = x_test_label[t_no].reshape(nx2, ny2)
    AError = np.absolute(xP - xG) 
    AError[AError >= 0.5] = 1.0
    AError[AError < 0.5] = 0.0
    ax = plt.imshow(np.rot90(AError[:,:]),cmap = 'gray', vmin = 0, vmax = 1.0)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('test'+str(i+1)+'_4.jpg', dpi=300, bbox_inches = "tight")
    plt.close()
    
    print('P:',p_test[t_no,0]*20+20)
    print('V:',v_test[t_no,0]*2.0+0.5)
    



