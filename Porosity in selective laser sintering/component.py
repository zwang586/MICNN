# -*- coding: utf-8 -*-
####Use the trained yNet to perform component-scale simulation
####NOTE: Require high-performance computer since generating large-scale raw powder bed is very slow.


import numpy as np
import matplotlib.pyplot as plt
from powderGeneration import *
from model import *
from calYmax import *

nx = 35500
ny = 315*70 #315 layers, each with a height of 70-pixel or 140 uum.
nx2 = 128
ny2 = 128
MICNN = yNet(nx2,ny2)
MICNN.summary()
MICNN.load_weights("weights_yNet.h5")
p_test = 40.0
v_test = 0.5
p_test = (p_test-20.0)/20.0
v_test = (v_test-0.5)/2.0  
v_test = np.reshape(v_test, (1, 1))
p_test = np.reshape(p_test, (1, 1))

#Load slicedCAD data
MNum = np.loadtxt('data_slicedCAD\\MLogo_num.txt')
MXstart = np.loadtxt('data_slicedCAD\\MLogo_xStart.txt')
MXend = np.loadtxt('data_slicedCAD\\MLogo_xEnd.txt')
MNum = MNum.astype(np.int)
MXstart = MXstart.astype(np.int)
MXend = MXend.astype(np.int)

rad_std = 0.1
eta = np.zeros((nx,ny))
eta1 = np.zeros((nx,ny))
ymax0 = np.zeros(nx)
ymax1 = np.zeros(nx)
nn = 0 #Index of MXstart, MXend files.
layerH = 35 #Height of a mini-powder bed, thus each long track is made of two layers of many mini-powder beds
layerSeg = 100 #Length of a mini-powde bed. 
for j in range(1,631):
    print("Generating powder bed:", (j+1)//2)
####################Generate a series of mini-powder beds to create a long track##########################################
    layerNo = j
    layerSegNo = int((nx-1)/layerSeg)+1
    if j ==1:
        for k in range(1,layerSegNo+1):
#            print(k)
            etaSeg = eta[(k-1)*100:k*100,(j-1)*layerH:j*layerH]
            ymax0Seg = ymax0[(k-1)*100:k*100]
            for i in range(0,1000):
                [eta1Seg,yH,ymax1Seg] = powder2(etaSeg,layerSeg,layerH,1,layerH,ymax0Seg,rad_std) #powderv2(s_eta,nx,ny,layer,t,ymax,rad_std): 
        #        print(yH)
                if yH >layerH*1: #break when the to-be-added particle higher than layerH*layerNo 
                    break
                etaSeg = eta1Seg
                ymax0Seg = ymax1Seg
            eta[(k-1)*100:k*100,(j-1)*layerH:j*layerH] = etaSeg 
    else:
        if j % 2 == 1: #3, 5, 7 ... layer
            for k in range(1,layerSegNo+1):
                etaSeg = eta[(k-1)*100:k*100,(j-1)*layerH-30:j*layerH]
                ymax0Seg = 30-((j-1)*layerH-ymax0[(k-1)*100:k*100])
                for i in range(0,1000):
                    [eta1Seg,yH,ymax1Seg] = powder2(etaSeg,layerSeg,layerH+30,1,layerH+30,ymax0Seg,rad_std) #powderv2(s_eta,nx,ny,layer,t,ymax,rad_std): 
                    if yH >layerH*1+30: #break when the to-be-added particle higher than layerH*layerNo 
                        break
                    etaSeg = eta1Seg
                    ymax0Seg = ymax1Seg
                eta[(k-1)*100:k*100,(j-1)*layerH-30:j*layerH] = etaSeg  
        else: #2, 4, 6 ... layer
            for k in range(1,layerSegNo+1+1):
#                print(k)
                if k == 1:                    
                    etaSeg = eta[0:50,(j-1)*layerH-30:j*layerH]
                    ymax0Seg = 30-((j-1)*layerH-ymax0[0:50])
                    for i in range(0,1000):
                        [eta1Seg,yH,ymax1Seg] = powder2(etaSeg,50,layerH+30,1,layerH+30,ymax0Seg,rad_std) #powderv2(s_eta,nx,ny,layer,t,ymax,rad_std): 
                        if yH >layerH*1+30: #break when the to-be-added particle higher than layerH*layerNo 
                            break
                        etaSeg = eta1Seg
                        ymax0Seg = ymax1Seg
                    eta[0:50,(j-1)*layerH-30:j*layerH] = etaSeg
                elif k == layerSegNo+1:
                    etaSeg = eta[(k-1)*100-50:nx,(j-1)*layerH-30:j*layerH]
                    ymax0Seg = 30-((j-1)*layerH-ymax0[(k-1)*100-50:nx])
                    for i in range(0,1000):
                        [eta1Seg,yH,ymax1Seg] = powder2(etaSeg,50,layerH+30,1,layerH+30,ymax0Seg,rad_std) #powderv2(s_eta,nx,ny,layer,t,ymax,rad_std): 
                        if yH >layerH*1+30: #break when the to-be-added particle higher than layerH*layerNo 
                            break
                        etaSeg = eta1Seg
                        ymax0Seg = ymax1Seg
                    eta[(k-1)*100-50:nx,(j-1)*layerH-30:j*layerH] = etaSeg  
                else:
                    etaSeg = eta[(k-1)*100-50:k*100-50,(j-1)*layerH-30:j*layerH]
                    ymax0Seg = 30-((j-1)*layerH-ymax0[(k-1)*100-50:k*100-50])
                    for i in range(0,1000):
                        [eta1Seg,yH,ymax1Seg] = powder2(etaSeg,layerSeg,layerH+30,1,layerH+30,ymax0Seg,rad_std) #powderv2(s_eta,nx,ny,layer,t,ymax,rad_std): 
                        if yH >layerH*1+30: #break when the to-be-added particle higher than layerH*layerNo 
                            break
                        etaSeg = eta1Seg
                        ymax0Seg = ymax1Seg
                    eta[(k-1)*100-50:k*100-50,(j-1)*layerH-30:j*layerH] = etaSeg  
                    


###########################yNet based sintering simulation###########################
    eta1 = eta
    if j % 2 == 0: #Perform sintering simulation every 2 layers of mini-powder beds.
        for iii in range(1,int(MNum[int(j/2)-1])+1):
            if j*layerH < 128: #first one or two layers, in which case the total height is less than 128
                nx3 = ((MXend[nn]-MXstart[nn]+127)//128)*128
                ny3 = 128
                x_test = np.zeros((nx3,ny3))
                y_test = np.zeros((nx3,ny3))
                x_test[0:int(MXend[nn]-MXstart[nn]),128-j*layerH:128] = eta[MXstart[nn]:MXend[nn],0:j*layerH]
                x_test = np.reshape(x_test, (1, nx3, ny3, 1)) 
                
                MICNN = yNet(nx3,ny3)
                MICNN.load_weights("weights_yNet.h5")
                
                decoded_imgs = MICNN.predict([x_test,v_test,p_test])
                y_test = decoded_imgs[0].reshape(nx3, ny3)
                eta1[MXstart[nn]:MXend[nn],0:j*layerH] = y_test[0:int(MXend[nn]-MXstart[nn]),(128-j*layerH):128]
            else: #total height larger than 128
                nx3 = ((MXend[nn]-MXstart[nn]+127)//128)*128
                ny3 = 128
                x_test = np.zeros((nx3,ny3))
                y_test = np.zeros((nx3,ny3))
                x_test[0:int(MXend[nn]-MXstart[nn]),:] = eta[MXstart[nn]:MXend[nn],j*layerH-128:j*layerH]
                x_test = np.reshape(x_test, (1, nx3, ny3, 1))  
                
                MICNN = yNet(nx3,ny3)
                MICNN.load_weights("weights_yNet.h5")
                
                decoded_imgs = MICNN.predict([x_test,v_test,p_test])
                y_test = decoded_imgs[0].reshape(nx3, ny3)
                eta1[MXstart[nn]:MXend[nn],j*layerH-128:j*layerH] = y_test[0:int(MXend[nn]-MXstart[nn]),:]   
            nn = nn + 1
        eta = eta1
        
      
    ymax0 = calYmax(eta,nx,ny)
    for iii in range(0,nx):
        if ymax0[iii] <= (j-1)*layerH:
            ymax0[iii] = (j-1)*layerH + 6
 
    if j == 2 or j%50 == 0: ##Save structure data every 25 layers
        np.save('eta_layer_'+str(int(j/2))+'.npy',eta)
        
##Save and plot final result        
np.save('eta_layer_final.npy',eta)
ax = plt.imshow(np.rot90(eta[:,:]), cmap='gray')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.savefig('eta_layer_final.jpg', dpi=5000, bbox_inches = "tight")  

