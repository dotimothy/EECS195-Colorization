# model.py Model generated using keras and tensorflow #
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang
from re import S
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os

#set image directory and dimensions to feed into network
train_data_dir = os.getcwd() + '\\' + 'train'
val_data_dir = os.getcwd() + '\\' + 'val'
img_width, img_height = 256, 256

#remove horizontal_flip if overtraining
train_datagen = ImageDataGenerator(shear_range = .2, zoom_range = .2, rotation_range = 20, horizontal_flip = True)
test_datagen = ImageDataGenerator()
traindata = train_datagen.flow_from_directory(directory=train_data_dir, target_size=(img_width, img_height))
testdata =  test_datagen.flow_from_directory(directory=val_data_dir, target_size=(img_width, img_height))

#input shape first entry may be varied depending on our results, higher batch sizes might realize better results

#start actual model
#convolutional layers
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)

encoder_output1 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output1 = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output1)
encoder_output1 = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output1)

encoder_output2 = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output1)
encoder_output2 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output2)

#Decoder
#reshape this first block
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output2)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model.compile(optimizer='rmsprop', loss='mse')
model.summary()




