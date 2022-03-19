# -*- coding: utf-8 -*-
# yNet and conventional multi-input CNN implemented in stress field development modeling.

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def yNet(nx2,ny2):  
    input_X = Input(shape = (nx2,ny2,1),name='Input1')  #Field input
    input_a = Input(shape=(1,),name = 'Input2')  # Condition parameter input
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
    pool4 = MaxPooling2D(pool_size=(2, 2), name = 'block4_pool')(conv4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block5_conv1')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block5_conv2')(conv5)
    drop5 = Dropout(0.5,name = 'block5_drop')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name = 'block5_pool')(drop5)
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block6_conv1')(pool5)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name = 'block6_conv2')(conv6)
    drop6 = Dropout(0.5,name = 'block6_drop')(conv6)
    
    #Expand input_a to a 1x512 embedding vector through MLP
    h1 = Dense(256, activation='sigmoid',name = 'fc1')(input_a)
    h2 = Dense(512, activation='sigmoid',name = 'fc2')(h1)
    
    #Signal merging by gating feature maps
    merge = Multiply(name = 'merge')([drop6,h2])
    
    #Decoder
    up1 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge))
    merge1 = concatenate([drop5,up1], axis = 3)
    deconv1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    deconv1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv1)
    
    up2 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv1))
    merge2 = concatenate([conv4,up2], axis = 3)
    deconv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    deconv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv2)
    
    up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv2))
    merge3 = concatenate([conv3,up3], axis = 3)
    deconv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    deconv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv3)
    
    up4 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv3))
    merge4 = concatenate([conv2,up4], axis = 3)
    deconv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    deconv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv4)
    
    up5 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(deconv4))
    merge5 = concatenate([conv1,up5], axis = 3)
    deconv5 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    deconv5 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv5)
    deconv5 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(deconv5)
    deconv5 = Conv2D(1, 1, activation = 'sigmoid')(deconv5)
    Y = deconv5
    
    model = Model([input_X,input_a], Y)
    
    return model
