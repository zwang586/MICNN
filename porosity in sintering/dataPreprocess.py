import numpy as np
nTrain = 2250
nTest = 750
nTotal = nTrain + nTest

eta = np.load('data\\eta.npy') #before sintering
eta1 = np.load('data\\eta1.npy') #after sintering
print(np.shape(eta))

p = np.zeros(58*nTotal,)
v = np.zeros(58*nTotal,)
loc = np.zeros(57*nTotal,)
p_data = np.loadtxt('data\\pTrain.txt')
v_data = np.loadtxt('data\\vTrain.txt') 
no = 0
for n in range(0,nTrain):
    for m in range(0,58):
        p[no] = p_data[n]
        v[no] = v_data[n]
        no = no+1
       
     
p_data = np.loadtxt('data\\pTest.txt')
v_data = np.loadtxt('data\\pTest.txt')     
no = 0      
for n in range(0,nTest):
    for m in range(0,58):
        p[no+57*nTrain] = p_data[n]
        v[no+57*nTrain] = v_data[n]  
        no = no+1
p = (p-20.0)/20.0
v = (v-0.5)/2.0    
 
nx = 128
ny = 128
no = 0
x_data = np.ndarray((58*nTotal,nx,ny),dtype = np.float32) #adapt dimension 1, if using different x increment during extraction and total number of structures
y_data = np.ndarray((58*nTotal,nx,ny),dtype = np.float32) #adapt dimension 1, if using different x increment during extraction and total number of structures
for n in range(0,nTotal):  #loop over structures
        for m in range(0,58): #loop along x direction in each structure
            x_data[no,:,:] = eta[n,(10+10*m):(10+10*m+ny),22:150]  #adapt upper_bound of y#starting from pixel = 10, increment = 10 pixels
            loc[no] = m*1.0/57.0
            no = no+1

no = 0
for n in range(0,nTotal):  #loop over structures
        for m in range(0,57): #loop along x direction in each structure
            y_data[no,:,:] = eta1[n,(10+10*m):(10+10*m+ny),22:150]  #adapt upper_bound of y#starting from pixel = 10, increment = 10 pixels
            no = no+1
            
