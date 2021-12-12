# -*- coding: utf-8 -*-
# yNet and conventional multi-input CNN implemented in fluid dynamics modeling.

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def yNet(nx2,ny2):  
    input_X = Input(shape = (nx2,ny2,1),name='Input1')  #Field input
    input_dt = Input(shape=(1,),name = 'Input2')
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block1_conv1')(input_X)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name = 'block1_pool')(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block2_conv1')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name = 'block2_pool')(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block3_conv1')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block3_conv2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name = 'block3_pool')(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block4_conv1')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block4_conv2')(conv4)
    drop4 = Dropout(0.5,name = 'block4_drop')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name = 'block4_pool')(drop4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block5_conv1')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block5_conv2')(conv5)
    drop5 = Dropout(0.5,name = 'block5_drop')(conv5)
    
    #Expand input_a to a 1x256 embedding vector through MLP
    h1 = Dense(128, activation='sigmoid',name = 'fc1')(input_dt)
    h2 = Dense(256, activation='sigmoid',name = 'fc2')(h1)
    
    #Signal merging by gating feature maps
    merge = Multiply(name = 'merge')([drop5,h2])
    
    #Decoder
    
    up1 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge))
    merge1 = concatenate([drop4,up1], axis = 3)
    deconv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    deconv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv1)
    
    up2 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv1))
    merge2 = concatenate([conv3,up2], axis = 3)
    deconv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    deconv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv2)
    
    up3 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv2))
    merge3 = concatenate([conv2,up3], axis = 3)
    deconv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    deconv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv3)
    
    up4 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv3))
    merge4 = concatenate([conv1,up4], axis = 3)
    deconv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    deconv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv4)
    deconv4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv4)
    deconv4 = Conv2D(1, 1, activation = 'sigmoid')(deconv4)
    Y = deconv4
    
    model = Model([input_X,input_dt], Y)
    
    return model
