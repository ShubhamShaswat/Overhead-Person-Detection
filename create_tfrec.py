import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#IMAGE_DIR = "../input/celeba-dataset/img_align_celeba/img_align_celeba"
IMG_HEIGHT = 128 
IMG_WIDTH =  128
BATCH_SIZE = 64
AUTO = tf.data.experimental.AUTOTUNE
file_dir = '../input/overhead-depth-images-people-detection-gotpd1/DPDnet_training/DPDnet_training/train/'

###The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):

  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
 return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_inp_string,image_out_string):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  image_inp_shape = tf.image.decode_jpeg(image_inp_string).shape
  image_out_shape = tf.image.decode_jpeg(image_out_string).shape
  feature = {

      'image_input': _bytes_feature(image_inp_string),
      'image_output':_bytes_feature(image_out_string),
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


  #--------------------------------------------------------------------------------------

  ###process image
import cv2
def process_img(img):
    
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)

  # crop image  
  #credit: https://www.kaggle.com/tobirohrer/gan-with-tensorflow-and-tf-dataset
  #img = tf.image.central_crop(img, 0.7)
  #img = tf.image.crop_to_bounding_box(img, 
                                    #offset_height = 30, 
                                    #offset_width = 10, 
                                    #target_height = 115, 
                                    #target_width = 115
                                     #)

  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.    
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

###function to create tfrecord
def create_tfrecord(file_dir,name='gotpd-training'):
    #tfrecord filename
    filename = name + '.tfrecord'
    input_file = os.path.join(file_dir,'inputs')
    output_file = os.path.join(file_dir,'outputs')
    
    with tf.io.TFRecordWriter(filename) as writer:
        for image in tqdm(os.listdir(input_file)):
            
            
            img1 = cv2.imread(os.path.join(input_file,image))
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) # Fix incorrect colors
            img_inp_string = cv2.imencode('.jpg', img1, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            
            img2 = cv2.imread(os.path.join(output_file,image))
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) # Fix incorrect colors
            img_out_string = cv2.imencode('.jpg', img2, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            #img_string = open(os.path.join(file_dir,image), 'rb').read()
    
            #img = process_img(img)
            #img_string = base64.b64encode(img)
                            
            example = serialize_example(img_inp_string,img_out_string)
            writer.write(example)

create_tfrecord(file_dir=file_dir)