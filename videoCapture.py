import time
import numpy
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

out_directory = '/home/shubham/Downloads/gotpd_test_outputs/seq-P05-M04-A0001-G03-C00-S0030'
inp_directory = '/home/shubham/Downloads/gotpd_test_inputs/seq-P05-M04-A0001-G03-C00-S0030'

inp_images = os.listdir(inp_directory)
out_images = os.listdir(out_directory)

inp_images.sort()
out_images.sort()

index = len(inp_images) // 2

img = Image.open(os.path.join(inp_directory,inp_images[index]))
labels = Image.open(os.path.join(out_directory,out_images[index]))

img = np.asarray(img)
labels = np.asarray(labels)

im = np.add(img,labels)

#z = np.zeros((424, 512, 1)) 
#g = im[:,:,1]
#g = np.expand_dims(g,axis=-1)
#new_im = np.concatenate((z,g,z),axis=-1)
#np.place(r, r == 255, 0.)
#max_pixel, min_pixel = np.max(im),np.min(im)
#print(max_pixel,min_pixel)
#plt.imshow(im)
#plt.show()





fig = plt.figure( 1 )
ax = fig.add_subplot( 111 )
ax.set_title("My Title")

im = ax.imshow( numpy.zeros( ( 424, 512, 3 ) ) ) # Blank starting image
fig.show()
im.axes.figure.canvas.draw()

tstart = time.time()
for i in range(len(inp_images)) :

  img = Image.open(os.path.join(inp_directory,inp_images[i]))
  labels = Image.open(os.path.join(out_directory,out_images[i]))

  img = np.asarray(img)
  labels = np.asarray(labels)

  data = np.add(img,labels)

  #ax.set_title( str( a ) )
  im.set_data( data )
  im.axes.figure.canvas.draw()

print ( 'FPS:', 100 / ( time.time() - tstart ) )
