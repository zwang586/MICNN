# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:00:46 2020

@author: zwang
"""
#Add one powder to input structure s_eta
#Return new structure with the added powder

import numpy as np
import matplotlib.pyplot as plt

    

def powder3(s_eta,nx,ny,layer,t,ymax):  #layer:layer_no, t: layer thickness.
    
    nyMin = int(t)*(layer-1)-15
    if nyMin < 0:
        nyMin = 0
    nyMax = int(t)*layer
    
    rad_mean = 10.00  #Multilayer validation: 10.0, otherwise 12.5
    rad_std = 1.25  #Multiplayer validation:1.25, otherwise 0.25
    radius00 = np.random.normal(rad_mean,rad_std)
    if radius00 < 2:
        radius00 = np.random.normal(rad_mean,rad_std)
#    print(radius00)
    
#    ymax = np.zeros(nx)
#    
#    for i in range(0,nx):
#        for j in range(nyMin,nyMax):
#            if (s_eta[i,j] > 0.9):
#                ymax[i] = j
    
    phi_top = np.zeros((nx,ny))
    for i in range(0,nx): #loop all ymax[i]
        for j in range( max( ([0,i-int(radius00)-2]) ) ,min( ([nx,i+int(radius00)+2]) )  ):
            for k in range( max( ([nyMin,int(ymax[i]-radius00)-2]) ) ,min( ([nyMax,int(ymax[i]+radius00)+2]) )  ):
                if ( (j-i)**2+(k-ymax[i])**2 <= radius00**2):
                    phi_top[j,k] = 1.0
#    ax = plt.imshow(phi_top,cmap = 'gray', aspect = "equal")
#    plt.show()                
    yc = np.zeros(nx)
    for i in range(int(radius00)-1,nx-int(radius00)-1):
        for j in range(nyMin,nyMax):
            if(phi_top[i,j] == 1.0):
                yc[i] = j
                
    yc00 = min(yc[int(radius00)-1:nx-int(radius00)-1])
    
    for i in range(1,nx):
        if (yc[i] == yc00):
            xc00 = i
    if (yc00+radius00 < t*layer):        
        for i in range(0,nx):
            for j in range(0,ny):
                if( (i-xc00)**2 + (j-yc00)**2 <= radius00**2):
                    s_eta[i,j] = 1.0
    ymax1 = ymax
    for i in range(max(0,int(xc00-radius00-2)),min(int(xc00+radius00+2),nx)):
        for j in range(nyMin,nyMax):
            if (s_eta[i,j] > 0.9):
                ymax1[i] = j
    
                        
    
    return(s_eta,yc00+radius00,ymax1)  

def powder3v2_5a(s_eta,nx,ny,layer,t,ymax,rad_std):  #layer:layer_no, t: layer thickness.
    
    nyMin = int(t)*(layer-1)-40
    if nyMin < 0:
        nyMin = 0
    nyMax = int(t)*layer
    
    rad_mean = 10.0
    radius00 = np.random.normal(rad_mean,rad_std)
    if radius00 < 2:
        radius00 = 2
        
#    if np.random.rand() < 0.50:
#        rad_mean = 10.0
#        radius00 = np.random.normal(rad_mean,rad_std)
#        if radius00 < 2:
#            radius00 = 2
#    else:
#        radius00 = 12.5*0.5  

    
#    ymax = np.zeros(nx)
#    
#    for i in range(0,nx):
#        for j in range(nyMin,nyMax):
#            if (s_eta[i,j] > 0.9):
#                ymax[i] = j
    
    phi_top = np.zeros((nx,ny))
    for i in range(0,nx): #loop all ymax[i]
        for j in range( max( ([0,i-int(radius00)-2]) ) ,min( ([nx,i+int(radius00)+2]) )  ):
            for k in range( max( ([nyMin,int(ymax[i]-radius00)-2]) ) ,min( ([nyMax,int(ymax[i]+radius00)+2]) )  ):
                if ( (j-i)**2+(k-ymax[i])**2 <= radius00**2):
                    phi_top[j,k] = 1.0
#    ax = plt.imshow(phi_top,cmap = 'gray', aspect = "equal")
#    plt.show()                
    yc = np.zeros(nx)
    for i in range(int(radius00)-1,nx-int(radius00)-1):
        for j in range(nyMin,nyMax):
            if(phi_top[i,j] == 1.0):
                yc[i] = j
                
    yc00 = min(yc[int(radius00)-1:nx-int(radius00)-1])
    
    for i in range(1,nx):
        if (yc[i] == yc00):
            xc00 = i
    if (yc00+radius00 < t*layer):        
        for i in range(0,nx):
            for j in range(0,ny):
                if( (i-xc00)**2 + (j-yc00)**2 <= radius00**2):
                    s_eta[i,j] = 1.0
    ymax1 = ymax
    for i in range(max(0,int(xc00-radius00-2)),min(int(xc00+radius00+2),nx)):
        for j in range(nyMin,nyMax):
            if (s_eta[i,j] > 0.9):
                ymax1[i] = j
    
                        
    
    return(s_eta,yc00+radius00,ymax1)      