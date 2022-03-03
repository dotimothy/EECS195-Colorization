from pyexpat import model
from re import S
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.models import Sequential, Model, load_model
import os
from matplotlib.image import imread
from keras.applications.vgg16 import VGG16
from time import time


model = load_model('kaggle.h5')
model.summary()