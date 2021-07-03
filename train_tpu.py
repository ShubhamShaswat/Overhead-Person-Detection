
import tensorflow as tf
import matplotlib.pyplot as plt
#from model import dpdnet 

from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, MaxPooling2D, LeakyReLU, Add,
    SeparableConv2D, Cropping2D, Conv2DTranspose, ZeroPadding2D
)
from tensorflow.keras import Model

##----------------------------------------------------------------------------DPDNET MDOEL----------------------------------------------------
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
def main_block(inp):
    #inp = Input(shape=shape)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = encoding_block(x,a=64,b=64,c=256,k=3,s=1)
    x = encoding_block(x,a=128,b=128,c=512,k=3,s=2)
    x = encoding_block(x,a=256,b=256,c=1024,k=3,s=2)

    x = decoder_block(x,a=1024,b=1024,c=256,k=3,s=1)

    x = decoder_block(x,a=512,b=512,c=128,k=3,s=2)
    x = decoder_block(x,a=256,b=256,c=64,k=3,s=2)
    x = Cropping2D([(0,0),(0,1)])(x)
    
    x = UpSampling2D((3,3))(x)
    
    x = Conv2DTranspose(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = Cropping2D([(2,2),(1,1)])(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(1, kernel_size=(3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)
    
    return x

###Main Block
#encoder 
def refinement_block(inp):
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

def dpdnet(shape=(212,256,1)):
    inp1 = Input(shape=shape)
    out1 = main_block(inp1)
    inp2 = tf.keras.layers.Concatenate(axis=-1)([inp1, out1])
    out2 = refinement_block(inp2)
    return Model(inp1,[out1,out2])

##------------------------------------------------------------------------------PREPARE DATASET----------------------------------------------------------------

IMAGE_SIZE = [212,256]
BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Number of accelerators: ", tpu_strategy.num_replicas_in_sync)


#get GSC path
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path('gotpdtfrecord')
print(GCS_DS_PATH)


image_feature_description = {

    'image_input': tf.io.FixedLenFeature([], tf.string),
    'image_output': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
  image_inp = decode_image(parsed_example['image_input'])
  image_out = decode_image(parsed_example['image_output'])
    
  return image_inp,image_out


def decode_image(image_data):
    #decode image 
    image = tf.image.decode_jpeg(image_data, channels=1)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,IMAGE_SIZE)
    return image

data_dir = GCS_DS_PATH + '/*.tfrecord'

def get_dataset(batch_size):
    filenames = tf.io.gfile.glob(data_dir)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(10000).batch(batch_size).prefetch(AUTO)
    return dataset

##-----------------------------------------------------------------------TPU TRAINING-----------------------------------------------------------------------
with tpu_strategy.scope():

        #define model
        model = dpdnet()
        #define optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        #define loss 
        loss_metric = tf.keras.metrics.Mean(name='mse')
        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    

    per_replica_batch_size =  BATCH_SIZE // tpu_strategy.num_replicas_in_sync
    train_dataset = tpu_strategy.experimental_distribute_datasets_from_function(lambda _:get_dataset(per_replica_batch_size))
    
#define train step
@tf.function
def train_step(iterator):
    
    def step_fn(x):
        """The computation to run on each TPU device."""
       
        with tf.GradientTape() as tape:
            #get loss
            output1,output2 = model(x[0], training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            loss_value1 = loss_fn(x[1],output1)
            loss_value2 = loss_fn(x[1],output2)
            loss = loss_value1 + loss_value2
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        loss_metric.update_state(loss * tpu_strategy.num_replicas_in_sync) # * tpu_strategy.num_replicas_in_sync 
    
    tpu_strategy.run(step_fn, args = (next(iterator),))

epochs = 2
steps_per_epoch = 2000
train_iterator = iter(train_dataset)
for epoch in range(epochs):
    
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step in range(steps_per_epoch):
        # Log every 200 batches.
        train_step(train_iterator)
        
        if step % 500 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_metric.result()))
            )
            print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
        loss_metric.reset_states()

##---------------------------------------------------------MODEL VALIDATION------------------------------------------------------------------------------------