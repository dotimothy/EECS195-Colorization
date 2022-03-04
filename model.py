import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import tensorflow as tf
from keras.layers import *
import os


IMG_WIDTH =  200    
IMG_HEIGHT = 200
batch_size = 1

train_dir = os.getcwd() + "\\train\\"
val_dir   = os.getcwd() +  "\\val\\"

image_gen_train = ImageDataGenerator(rescale=1./255, 
                                     zoom_range=0.2, 
                                     rotation_range=65,
                                     shear_range=0.09,
                                     horizontal_flip=True,
                                     vertical_flip=True)
image_gen_val = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,directory=train_dir,
shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='sparse')
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
directory=val_dir,target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='sparse')
model=tf.keras.Sequential(
        [
            InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            Conv2D(filters=64, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=64, kernel_size=3, strides= 2, activation='relu'),
            Conv2D(filters=64, kernel_size=3, strides= 1, activation='relu'),
            UpSampling2D(data_format = 'channels_first', interpolation = 'bilinear'),
            
            
            Conv2D(filters=128, kernel_size=3, strides= 2, activation='relu'),
            Conv2D(filters=128, kernel_size=3, strides= 1, activation='relu'),
            UpSampling2D(data_format = 'channels_first', interpolation = 'bilinear'),
            
            Conv2D(filters=256, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=256, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=256, kernel_size=3, strides= 1, activation='relu'),
            UpSampling2D(data_format = 'channels_first', interpolation = 'bilinear'),
            
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            UpSampling2D(data_format = 'channels_first', interpolation = 'bilinear'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            Conv2D(filters=512, kernel_size=3, strides= 1, activation='relu'),
            UpSampling2D(data_format = 'channels_first', interpolation = 'bilinear'),
            
            
            
            
            Flatten()
            
            
         ])
#Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
#Fitting the model
history = model.fit(train_data_gen,steps_per_epoch=len(train_data_gen)//batch_size, validation_data=val_data_gen, epochs=20)
model.save('model.h5')
model.summary()