#encoder module

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU
)

###Main Block
#encoder 
def encoder(shape=(212,256,1)):

	inp = Input(shape=shape)
	x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	#Encoding Conv Block
	x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	



