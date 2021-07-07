import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

data_dir = '/home/shubham/Downloads/gotpd_test_inputs/seq-P05-M04-A0001-G03-C00-S0030/'
print(os.getcwd())
infer = tf.keras.models.load_model('july')
print('MODEL LOADED...')
#print(f)


##load images from the directory as a numpy array
img_seq = []
for image in os.listdir(data_dir):
  img = tf.io.read_file(os.path.join(data_dir,image))
  img = tf.image.decode_jpeg(img, channels=1)
  img = tf.cast(img, tf.float32) / 255.
  img = tf.image.resize(img, [212,256])
  img_seq.append(img)

img_seq = np.stack(img_seq, axis=0)
print(img_seq.shape)





##Below deosn't work with save mdoel
"""
def decode_image(filename):
    #decode image
    bits = tf.io.read_file(filename) 
    image = tf.image.decode_jpeg(bits,channels=1)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,[212,256])
    return image

val_ds = tf.data.Dataset.list_files(data_dir)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  label_mode=None,
  image_size=(212, 256),
  batch_size=8)
val_ds = val_ds.map(decode_image).batch(10)
print(val_ds)

"""

#predicted = infer(val_ds)
#print('PREDICTION COMPLETE....')
#print(predicted.shape)
"""


plt.figure(figsize=(10, 10))
for images in val_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy())
   # plt.title(class_names[labels[i]])
    plt.axis("off")
  plt.show()
""" 