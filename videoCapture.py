import time
import numpy
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
directory = '/home/shubham/Downloads/gotpd_test-inputs/seq-P05-M04-A0001-G03-C00-S0030'

fig = plt.figure( 1 )
ax = fig.add_subplot( 111 )
ax.set_title("My Title")

im = ax.imshow( numpy.zeros( ( 512, 424, 3 ) ) ) # Blank starting image
fig.show()
im.axes.figure.canvas.draw()

tstart = time.time()
for a in os.listdir(directory):
  data = Image.open(directory + '/' + a)
  data = np.asarray(data)
                                         #numpy.random.random( ( 256, 256, 3 ) ) # Random image to display
  ax.set_title( str( a ) )
  im.set_data( data )
  im.axes.figure.canvas.draw()

print ( 'FPS:', 100 / ( time.time() - tstart ) )