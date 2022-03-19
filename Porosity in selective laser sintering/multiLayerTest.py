# -*- coding: utf-8 -*-

##Test the trained yNet in modeling multi-layer of long tracks under 3 different laser power conditions.
import numpy as np
import matplotlib.pyplot as plt
from model import *
from powderGeneration import *
from calYmax import *

#####Read and plot physical simulation results##########################
nx = 640
ny = 305
eta_final = np.zeros((3,nx,ny))
for i in range(0,3):
    filename = 'data_multiLayerTest\\case_'+str(int(1+i))+'_eta_final_3.txt'
    with open(filename) as f:
        lines = f.readlines()
    count = 0
    for j in range(0,ny):
        for k in range(0,nx):
            eta_final[i,k,j] = lines[count]
            count = count + 1
for i in range(0,3):
    ax = plt.imshow(np.rot90(eta_final[i,:,0:225]), cmap='gray', vmin = 0, vmax = 1)
#    plt.show()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig('mutliTest_physical_case_'+str(int(1+i))+'_eta_final_3.jpg', dpi=200, bbox_inches = "tight")
    plt.close()



####Produce and plot yNet results######################################  
nx = 640
ny = 320
nx2 = 640
ny2 = 128
thickness = 70

MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")
for nn in range(0,3):
    eta = np.zeros((nx,ny))
    eta1 = np.zeros((nx,ny))
    ymax0 = np.zeros(nx)
    p_test = 20+nn*5
    print("Case: Power = ",p_test)
    v_test = 0.5
    
    p_test = (p_test-20.0)/20.0
    v_test = (v_test-0.5)/2.0  
    
    v_test = np.reshape(v_test, (1, 1))
    p_test = np.reshape(p_test, (1, 1))
    
    
  
    for layer in range(1,4):
        ymax0 = calYmax(eta,nx,ny)
        for i in range(0,1000): ##Generate raw powder bed  
            [eta1,yH,ymax1] = powder(eta,nx,ny,layer,thickness,ymax0)
            print("y_newPowder:",yH)
            if yH >thickness*layer:
                break
            eta = eta1
            ymax0 = ymax1
        
            
            
        ax = plt.imshow(np.rot90(eta[:,0:225]), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('mutliTest_yNet_case_'+str(int(nn+1))+'_eta_0_'+str(layer)+'.jpg', dpi=200, bbox_inches = "tight")       
        
        if layer <= 1:
            x_test = np.zeros((nx2,ny2))
            x_test[:,128-(layer*thickness):128] = eta[:,0:layer*thickness]
            x_test = np.reshape(x_test, (1, nx2, ny2, 1))
            decoded_imgs = MICNN.predict([x_test,v_test,p_test])
            
            y_test = decoded_imgs[0].reshape(nx2, ny2)
            eta[:,0:layer*thickness] = y_test[:,128-(layer*thickness):128]
        
                                    
        else:
            x_test = np.zeros((nx2,ny2))
            x_test = eta[:,layer*thickness-128:layer*thickness]
            x_test = np.reshape(x_test, (1, nx2, ny2, 1))
            decoded_imgs = MICNN.predict([x_test,v_test,p_test])
            
            y_test = decoded_imgs[0].reshape(nx2, ny2)
            eta[:,layer*thickness-128:layer*thickness] = y_test        
            
        ax = plt.imshow(np.rot90(eta[:,0:225]), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('mutliTest_yNet_case_'+str(int(nn+1))+'_eta_final_'+str(layer)+'.jpg', dpi=200, bbox_inches = "tight")
 
        

