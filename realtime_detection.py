import cv2
import os
import time

INPUT_SRC = '/home/shubham/Downloads/gotpd_test_inputs/seq-P05-M04-A0001-G03-C00-S0030'
OUTPUT_SRC = os.path.join(os.getcwd(),'video.avi')
height = 424 
width =  512


tstart = time.time() #start the timer

IMAGES_SEQ = os.listdir(INPUT_SRC)
IMAGES_SEQ.sort()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video=cv2.VideoWriter(OUTPUT_SRC,fourcc,30,(width,height))

#Start Capturing freames
for image in IMAGES_SEQ:
    img = cv2.imread(os.path.join(INPUT_SRC,image))
    video.write(img)
cv2.destroyAllWindows()
video.release()

print ( 'TIME ELASPED: ' ,( time.time() - tstart ) ) #stop the timer