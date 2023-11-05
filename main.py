import cv2 
from ffpyplayer.player import MediaPlayer
import numpy as np
import os
import random

#Sources
source= 'video.mp4'

#Instances of librarys
cap = cv2.VideoCapture(source)
player = MediaPlayer(source)
frame_count = 0
#Random Variables
random_degree = random.uniform(-3.0, 3.0)
scale_size = random.uniform(0.9, 0.97)
random_brightness= random.uniform(0.8, 1.3)
random_contrast = random.uniform(0.01, 0.05)
random_flip = random.choice([0, 1])
random_crop = random.randint(9, 13)
random_pixel_coordinate_x = np.random.randint(0, 400, 10)
random_pixel_coordinate_y = np.random.randint(0, 400, 10)






def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1])/ 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def zoom(img, zoom_factor):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

while(cap.isOpened()==True):
    ret, frame = cap.read()
    if ret == True :
        frame = rotate_image(frame, random_degree)
        frame = zoom(frame, scale_size)
        frame =cv2.convertScaleAbs(frame,random_contrast, random_brightness)
        if random_flip == 0:
            frame = cv2.flip(frame, 1)
        for coordinate in random_pixel_coordinate_x:
            frame[random_pixel_coordinate_x, random_pixel_coordinate_y] = (255, 255, 255)
        frame = frame[random_crop:-random_crop, random_crop:-random_crop]
        cv2.imshow('Frame', frame)
        if(cv2.waitKey(25)== ord('q')):
            break
    else:
        break

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'XVID' or other codecs if necessary
print(fourcc)
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('H', '2', '6', '4'), 10, (400,400))

cap.release()
cv2.destroyAllWindows()

