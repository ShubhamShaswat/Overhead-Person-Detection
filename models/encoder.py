#encoder module

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU, Add
)

#Encoding Conv Block
def encoding_block(x,a,b,c,k):
	y = Conv2D(a, kernel_size=(1, 1), strides=(2, 2), padding='same')(y)
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	y = Conv2D(b, kernel_size=(k, k), strides=(1, 1), padding='same')(y)
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	y = Conv2D(c, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	
	y_shortcut = Conv2D(c, kernel_size=(1 1), strides=(2, 2), padding='same')(x)
	y_shortcut = BatchNormalization()(y_shortcut)
	y_out = Add()([y_shortcut,y])
	y_out = LeakyReLU(y_out)

	return y_out


###Main Block
#encoder 
def encoder(shape=(212,256,1)):

	inp = Input(shape=shape)
	x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)



