# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#############Process vm_stress of different shapes###################
##read vm-stress data in .txt file from COMSOL
nx = 128
ny = 128
num = 6000 ##total number of cases/txt files

#vmStress = np.zeros((num,nx*ny))
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
        x = lineSplit[2]
#######Extract input structure and output stressHotSpot           
        if x == "NaN":
            struct[i,j] = 0.0
            vmStress[i,j] = 0.0
        else:
            struct[i,j] = 1.0
            vmStress[i,j] = x

            
            
                
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