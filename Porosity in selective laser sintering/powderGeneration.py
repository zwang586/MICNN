# -*- coding: utf-8 -*-

##Add one powder to the lowest point of input structure - s_eta
##Return new structure with the added powder and other updated infomation.
 

import numpy as np
import matplotlib.pyplot as plt
   
def powder(s_eta,nx,ny,layer,t,ymax):  
#s_eta: input structure.
#layer:layer_no.
#t: layer thickness. 
#ymax: surface of s_eta structure.
    
    nyMin = int(t)*(layer-1)-15
    if nyMin < 0:
        nyMin = 0
    nyMax = int(t)*layer
    
    rad_mean = 12.5 
    rad_std = 1.25  
    radius00 = np.random.normal(rad_mean,rad_std)
    if radius00 < 2:
        radius00 = np.random.normal(rad_mean,rad_std)

    
    phi_top = np.zeros((nx,ny))
    for i in range(0,nx): #loop pver all ymax[i]
        for j in range( max( ([0,i-int(radius00)-2]) ) ,min( ([nx,i+int(radius00)+2]) )  ):
            for k in range( max( ([nyMin,int(ymax[i]-radius00)-2]) ) ,min( ([nyMax,int(ymax[i]+radius00)+2]) )  ):
                if ( (j-i)**2+(k-ymax[i])**2 <= radius00**2):
                    phi_top[j,k] = 1.0
           
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

def powder2(s_eta,nx,ny,layer,t,ymax,rad_std):  ##For use in full-component sintering simulation. Include powder radius_std as input.
    
    nyMin = int(t)*(layer-1)-40
    if nyMin < 0:
        nyMin = 0
    nyMax = int(t)*layer
    
    rad_mean = 6.5
    radius00 = np.random.normal(rad_mean,rad_std)
    if radius00 < 2:
        radius00 = 2
    
    phi_top = np.zeros((nx,ny))
    for i in range(0,nx): #loop over all ymax[i]
        for j in range( max( ([0,i-int(radius00)-2]) ) ,min( ([nx,i+int(radius00)+2]) )  ):
            for k in range( max( ([nyMin,int(ymax[i]-radius00)-2]) ) ,min( ([nyMax,int(ymax[i]+radius00)+2]) )  ):
                if ( (j-i)**2+(k-ymax[i])**2 <= radius00**2):
                    phi_top[j,k] = 1.0
           
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
