# MICNN: Multi-input convolutional network

Code used in paper “Multi-input convolutional network for ultrafast simulation of field evolvement”.

Multi-input CNN model code (model.py) is partly adapted from [Implementation of UNet](https://github.com/zhixuhao/unet).


## Dependencies

  Tensorflow 2.1.3 and Python 3.6.
  
## Data

Accompanied datasets can be downloaded from Mendeley Data: XX

## How to use

### Fluid dynamics

1. Run **dataPreprocess160.py** to prreprocess 30 groups of raw vorticity data under //data folder.
2. Run **trainYNet.py** to train and test the proposed multi-input CNN, yNet, for fluid dynamics modeling.
3. Run **dataPreprocess192.py** to preprocess 5 groups of raw vorticity data under //data folder.
4. Run **recurrent.py** to perform dynamic flow simulation using the trained yNet.
5. Run **trainOldMICNN.py** to train the conventional multi-input CNN.
6. Run **comparision.py** to see the comparision of yNet and conventional multi-input CNN at a random testing point.

### Porosity in selective laser sintering

1. Run **main.py** to train and test yNet for porosity modeling in selective laser sintering.


