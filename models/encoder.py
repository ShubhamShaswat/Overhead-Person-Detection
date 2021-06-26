#encoder module

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU, Add
)
from tensorflow.keras import Model

#Encoding Conv Block
def encoding_block(x,a,b,c,k):
	y = Conv2D(a, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	y = Conv2D(b, kernel_size=(k, k), strides=(1, 1), padding='same')(y)
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	y = Conv2D(c, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	
	y_shortcut = Conv2D(c, kernel_size=(1,1), strides=(2, 2), padding='same')(x)
	y_shortcut = BatchNormalization()(y_shortcut)
	y_out = Add()([y_shortcut,y])
	y_out = LeakyReLU()(y_out)

	return y_out


###Main Block
#encoder 
def encoder(shape=(212,256,1)):

	inp = Input(shape=shape)
	x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) #doubt about strides value

	x = encoding_block(x,a=64,b=64,c=256,k=3)
	x = encoding_block(x,a=128,b=128,c=512,k=3)
	x = encoding_block(x,a=256,b=256,c=1024,k=3)

	return Model(inp,x)





