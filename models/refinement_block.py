#encoder module

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU, Add,
    SeparableConv2D, Cropping2D, Conv2DTranspose, ZeroPadding2D
)
from tensorflow.keras import Model
#Encoding Conv Block
def encoding_block(x,a,b,c,k,s):
    y = SeparableConv2D(a, kernel_size=(1, 1), strides=(s, s), padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    y = SeparableConv2D(b, kernel_size=(k, k), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    y = SeparableConv2D(c, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    
    y_shortcut = SeparableConv2D(c, kernel_size=(1, 1), strides=(s, s))(x)
    y_shortcut = BatchNormalization()(y_shortcut)
    y_out = Add()([y_shortcut,y])
    y_out = LeakyReLU()(y_out)

    return y_out

#Decoding Conv Block
def decoder_block(x,a,b,c,k,s):
    y = UpSampling2D((s, s))(x)
    y = SeparableConv2D(a, kernel_size=(1, 1))(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    
    y = SeparableConv2D(b, kernel_size=(k, k), padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    
    y = SeparableConv2D(c, kernel_size=(1, 1))(y)
    y = BatchNormalization()(y)
    
    y_shortcut = UpSampling2D((s, s))(x)
    y_shortcut = SeparableConv2D(c, kernel_size=(1, 1))(y_shortcut)
    y_shortcut = BatchNormalization()(y_shortcut)
    y_out = Add()([y_shortcut,y])
    y_out = LeakyReLU()(y_out)
    return y_out

###Main Block
#encoder 
def refinement(inp):
    #inp = Input(shape=shape)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = encoding_block(x,a=64,b=64,c=256,k=3,s=1)
    x = encoding_block(x,a=128,b=128,c=512,k=3,s=2)
   
    x = decoder_block(x,a=512,b=512,c=128,k=3,s=2)
    x = decoder_block(x,a=256,b=256,c=64,k=3,s=2)

    x = ZeroPadding2D(padding=(0,1))(x)    
    x = UpSampling2D((3,3))(x)
    x = Cropping2D([(2,2),(1,1)])(x)
    x = Conv2DTranspose(1, kernel_size=(3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)
    
    return x