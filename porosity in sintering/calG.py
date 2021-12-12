# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:54:19 2020

@author: zwang
"""

#calculate pixel-based global accuracy
import numpy as np

def calG(s_eta_true,s_eta_pre,nx,ny,n_sample):
    for i in range(0,n_sample):
        for j in range(0,nx):
            for k in range(0,ny):
                if s_eta_true[i,j,k] >=0.5:
                    s_eta_true[i,j,k] = 1.0
                elif s_eta_true[i,j,k] <=0.5:
                    s_eta_true[i,j,k] = 0.0        

    for i in range(0,n_sample):
        for j in range(0,nx):
            for k in range(0,ny):
                if s_eta_pre[i,j,k] >=0.5:
                    s_eta_pre[i,j,k] = 1.0
                elif s_eta_pre[i,j,k] <=0.5:
                    s_eta_pre[i,j,k] = 0.0  

    nAcc = np.zeros(n_sample,)
    
    for i in range(0,n_sample):
        count = 0
        for j in range(0,nx):
            for k in range(0,ny):
                if s_eta_true[i,j,k] == s_eta_pre[i,j,k]:
                    count = count+1.0
        
        nAcc[i] = count/(nx*ny)
    
    mAcc = np.mean(nAcc)
    return mAcc