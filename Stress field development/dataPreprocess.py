# -*- coding: utf-8 -*-

import numpy as np


#############Process vm_stress of 7000 different perforation structures with ellicptical holes###################
##read vm-stress data in .txt file 
nx = 128
ny = 128
num = 7000 ##total number of cases/txt files


struct = np.zeros((num,nx*ny))   #strcutre
vmStress = np.zeros((num,nx*ny))   #vmStress Hop Spot
for i in range(0,num):
    print(i)
    filename = 'data\\'+'vmStress_'+str(i+1)+'.txt'
    with open(filename) as f:
        vmData = f.readlines()
    for j in range(0,len(vmData)):
        line = vmData[j]
        lineSplit = line.split()
        sigma = lineSplit[2]
        
##Extract input structure and output vmStress field          
        if sigma == "NaN": #Data inside hole is NaN
            struct[i,j] = 0.0
            vmStress[i,j] = 0.0
        else:
            struct[i,j] = 1.0
            vmStress[i,j] = sigma

                     
##Reshape raw data to 2D                
struct2D = np.zeros((num,nx,ny))
vmStress2D = np.zeros((num,nx,ny))
for k in range(0,num):
    count = 0
    for i in range(0,ny):
        for j in range(0,nx):
            struct2D[k,j,i] = struct[k,count]
            vmStress2D[k,j,i] = vmStress[k,count]
            count = count + 1

np.save('struct2D.npy',struct2D)
np.save('vmStress2D.npy',vmStress2D)            