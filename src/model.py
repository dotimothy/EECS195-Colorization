import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
from tensorflow import keras
from keras.layers import *
from keras import Sequential
from keras.models import load_model
import os

stopper = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 3, mode = "min")
checkpoints = tf.keras.callbacks.ModelCheckpoint(monitor = "val_accuracy",
                                                      save_best_only = True,
                                                      save_weights_only = True,
                                                      filepath = os.getcwd() + "./modelcheck")
encoder = Sequential()
encoder.add(Conv2D(filters = 64, kernel_size = 2 , padding = "same", use_bias = True, strides = 1))
# encoder.add(Conv2D(filters = 64, kernel_size = 2 , padding = "same", use_bias = True, strides = 1))
# encoder.add(Conv2D(filters = 64, kernel_size = 2 , padding = "same", use_bias = True, strides = 1))
encoder.add(BatchNormalization())
encoder.add(ReLU())

encoder.add(Conv2D(64, kernel_size = 2, padding = "same", use_bias = True))
encoder.add(BatchNormalization())
encoder.add(ReLU())

encoder.add(Conv2D(128, kernel_size = 2, padding = "same", use_bias = True))
encoder.add(BatchNormalization())
encoder.add(ReLU())

encoder.add(Conv2D(256, kernel_size = 2, padding = "same", use_bias = True))
encoder.add(BatchNormalization())
encoder.add(ReLU())


decoder = Sequential()
decoder.add(Conv2DTranspose(128, kernel_size = 2, padding = "same", use_bias = True))
decoder.add(ReLU())

decoder.add(Conv2DTranspose(64, kernel_size = 2, padding = "same", use_bias = True))

decoder.add(ReLU())

decoder.add(Conv2DTranspose(32, kernel_size = 2, padding = "same", use_bias = True))
decoder.add(ReLU())

decoder.add(Conv2DTranspose(3, kernel_size = 2, padding = "same", use_bias = True))
decoder.add(ReLU())

auto_encoder = Sequential([encoder, decoder])