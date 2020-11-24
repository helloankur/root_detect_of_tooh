import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.keras.models import Sequential
from matplotlib.image import imread
import os
train_dir='Main_data'
print(train_dir)
train_dir_class=(os.listdir(train_dir))
root_1=(os.listdir(os.path.join(train_dir,train_dir_class[0])))

root_2=(os.listdir(os.path.join(train_dir,train_dir_class[1])))
print(root_1)

for img1 in root_1:
    print(os.path.join(train_dir, train_dir_class[0], img1))

print(root_2)
for img2 in root_2:
    print(os.path.join(train_dir, train_dir_class[1], img2))

image_shape=(130,130,3)


from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img,load_img

image_gen=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.10,
                             height_shift_range=0.10,
                             rescale=1/255,
                             shear_range=0.1,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')


def gen_img_1():
    for img1 in root_1:
        img_path=(os.path.join(train_dir, train_dir_class[0], img1))
       # print(os.path.join(train_dir, train_dir_class[0], img1))

        img1_load=load_img(img_path)
        x=img_to_array(img1_load)
        x=x.reshape((1,)+x.shape)
        #print(x)
        i=0
        for batch in image_gen.flow(x,batch_size=1,save_to_dir='A_data',save_prefix="tooth_1",save_format="jpeg"):
            i +=1
            if i>20:
                break

def gen_img_2():
    for img2 in root_2:
        img_path=(os.path.join(train_dir, train_dir_class[1], img2))
       # print(os.path.join(train_dir, train_dir_class[0], img1))

        img2_load=load_img(img_path)
        x2=img_to_array(img2_load)
        x2=x2.reshape((1,)+x2.shape)
        #print(x2)
        i=0
        for batch in image_gen.flow(x2,batch_size=1,save_to_dir='B_data',save_prefix="tooth_2",save_format="jpeg"):
            i +=1
            if i>20:
                break



gen_img_1()
gen_img_2()





