#encoder module

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU
)


##encoder 

def encoder():

	inp = Input(shape=shape)
	x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)
