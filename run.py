import numpy as np
from nnet.network import Sequential
from nnet.layers import conv2d, BatchNormalization, dense, max_pool, dropout
from nnet import optimizers
from nnet import functions


model = Sequential()
model.add(conv2d(num_kernels = 128, kernel_size = 5, activation = functions.relu, input_shape = (128, 128, 5)))
model.add(BatchNormalization())
model.add(max_pool())
model.add(dropout(0.1))