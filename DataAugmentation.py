# !/usr/bin/python

""" Dataset Structure:
Data:
.....1_Images
..........1_001.jpeg
..........1_002.jpeg
..........1_003.jpeg

.
.
.

.....6_Images
..........6_001.jpeg
..........6_002.jpeg
..........6_003.jpeg


"""

import numpy 
import tensorflow as tf
from glob import glob
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

COUNTER = 10

DataGen = ImageDataGenerator(
    rotation_range = 40,width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True,vertical_flip = True,)


for k in range(1,7):
    imagelist = glob('Data/{}_Images/{}*'.format(k,k)) #Arrange all images in a class to a particular folder

    for eachImage in imagelist:
        img = load_img(eachImage,color_mode = 'grayscale')
        x = img_to_array(img)
        x = numpy.expand_dims(x,0)


        i = 0
        for batch in DataGen.flow(x,batch_size = 1, save_to_dir = 'Train/{}_Images'.format(k),save_prefix = str(k),save_format = 'jpeg'):
            i+=1
            if i>COUNTER:
                break
