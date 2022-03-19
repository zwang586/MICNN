# -*- coding: utf-8 -*-

#calculate pixel-based global accuracy
import numpy as np

def calGAcc(s_eta_true,s_eta_pre):
    s_eta_true = 1*(s_eta_true >=0.5)   #Binarization   
    s_eta_pre = 1*(s_eta_pre >=0.5)   #Binarization 
    
    n_sample = len(s_eta_true)
    nAcc = np.zeros(n_sample,) 
    
    for i in range(0,n_sample):  #Calculate global accuracy for each instance.
        nAcc[i] = np.mean(1*np.equal(s_eta_true[i,:,:],s_eta_pre[i,:,:]))
    
    meanAcc = np.mean(nAcc)
    return meanAcc