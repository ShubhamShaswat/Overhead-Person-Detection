import numpy as np
import matplotlib.pyplot as plt
import cv2
TEST_IMG = '/home/shubham/Downloads/gotpd_test_outputs/seq-P05-M04-A0001-G03-C00-S0030/image0126.png'

#function to calculate number of people and x,y cordinates

def func1(x,y):
    array_size = len(x)
    idx = []
    for i in range(array_size-1):
        d = (x[i]-x[i+1])**2 + (y[i]-y[i+1])**2
        d = np.sqrt(d)
        #print(d)
        if d < 2.0:
            idx.append(i)
            
    x = np.delete(x,idx)
    y = np.delete(y,idx)
    n = len(x)
    return x,y,n
        
            


img = cv2.imread(TEST_IMG,0)
filters = np.ones(img.shape) * 255.
print(np.max(img))
print(img[1].shape)
y, x = np.where(img == 255.)
x_,y_,n = func1(x,y)

print("Number of People : ", n)
plt.scatter(x_,y_)
plt.imshow(img)

plt.show()
