# model.py Model generated using keras and tensorflow #
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os

#set image directory and dimensions to feed into network
train_data_dir = os.getcwd() + '\\' + 'train'
val_data_dir = os.getcwd() + '\\' + 'val'
img_width, img_height = 224, 224

train_datagen = ImageDataGenerator()


traindata = train_datagen.flow_from_directory(directory=train_data_dir, target_size=(img_width, img_height))

#start actual model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (None, None, 1)))


#crf layer
#model.add(tf.layers.CRF())

#convolutional layers
model.add(tf.keras.layers.Conv2D())


