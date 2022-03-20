# -*- coding: utf-8 -*-

import numpy as np
def calYmax(s_eta,nx,ny):
    ymax = np.zeros(nx)
    
    for i in range(0,nx):
        for j in range(0,ny):
            if (s_eta[i,j] > 0.9):
                ymax[i] = j
                
    return ymax
    
