# MICNN: Multi-input convolutional network

Code used in paper “Multi-input convolutional network for ultrafast simulation of field evolvement”.

Multi-input CNN model code (model.py) is partly adapted from [Implementation of UNet](https://github.com/zhixuhao/unet).


## Dependencies

  Tensorflow 2.1.3 and Python 3.6.
  
## Data

Accompanied datasets can be downloaded from Mendeley Data: XX

Datasets were produced from massive (multi-)physics simulations. They are used to train multi-input convolutional network, which then can act as a cheap substitute of original physics-based models and allows for ultrafast simulation. 

The datasets and four related physical and engineering problems have distinct characteristics, which should present different challenges to a multi-input ConvNet. They can help comprehensively test the modeling capability of a multi-input ConvNet.

Note that the data requires further processing, namely properly preparing multi-input-output pairs, i.e.,((a,X), Y), for training the multi-input convolutional network. Please see the paper and code for greater details on how to use the data.

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
2. Run **singleLayerTest.py** to test the trained yNet on modeling single-layer small pataches and long tracks under 5 different laser power conditions.
3. Run **multiLayerTest.py** to test the trained yNet on modeling multi-layer long tracks under 3 different laser power conditions.
4. Run **component.py** to perform full-component (315 layers) selective laser sintering simulation using the trained yNet.

### Stress field development for perforation structures

1. Run **dataPreprocess.py** to preprocess 7000 stress field data under //data folder.
2. Run **main.py** to train and test yNet for modeling stress field development in perforation structure with elliptic holes.
3. Run **dataPreprocessExtrapolation.py** to preprocess 10 stress field data under //data_shapeExtrapolation folder.
4. Run **shapeExtrapolation.py** to test the trained yNet for modeling stress field development in perforation structure with other types of holes.

### Grain growth

1. Run **dataPreprocess.py** to preprocess grain growth simulation data under //data folder.
2. Run **main.py** to train and test yNet for grain growth simulation.
3. Run **largeScaleGG.py** to perform large-scale dynamic grain growth simulation using the initial structures under //data_seeding_1600x1600 and the trained yNet.


For more details on how to preprocess data, train and test multi-input convolutional network, and reproduce the results, please refer to the paper.


