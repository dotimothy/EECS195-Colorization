import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
from tensorflow import keras
from keras.layers import *
from keras import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from PIL import Image
from model import auto_encoder, stopper, checkpoints
import warnings
warnings.filterwarnings("ignore")


gray = np.load(os.getcwd() + "\\l_in.npy")
color = np.load(os.getcwd() + "\\ab_in.npy")

#pres_img = pres_img.astype("uint8").reshape(224, 224, 3)
splitting_count = 250
end_splitting_count = 500

x_train = gray[:splitting_count, :, :].astype("uint8").reshape(splitting_count, gray.shape[1], gray.shape[2])

y_train = color[:splitting_count, :, :].astype("uint8")

x_test = gray[splitting_count:end_splitting_count, :, :].astype("uint8").reshape(splitting_count, gray.shape[1], gray.shape[2])

y_test = color[splitting_count:end_splitting_count, :, :].astype("uint8")

def line_image(gray_images,splitting_count=splitting_count):
    zeros = np.zeros((splitting_count, 224, 224, 3))
    for i in range(0,3):
        zeros[:splitting_count,:,:, i] = gray_images[:splitting_count]
    return zeros

input_images = line_image(x_train, splitting_count)


def reconstruct(gray_images, ab_images, n):
    zeros = np.zeros((n, 224, 224, 3))
    zeros[:, :, :, 0] = gray_images[0:n:]
    zeros[:, :, :, 1:] = ab_images[0:n:]
    zeros = zeros.astype("uint8")
    img = []
    for i in range(0, n):
        img.append(zeros[i])
    img = np.array(img)
    return img

output_images = reconstruct(x_train, y_train, n = splitting_count)

gray_zeros = np.zeros((splitting_count,gray.shape[1],gray.shape[2], 1))
gray_zeros[:,:,:,0] = x_train

test_input = line_image(x_test, splitting_count)

test_output = reconstruct(x_test, y_test, n = end_splitting_count - splitting_count)

compile_loss = "mse"
compile_optimizer = Adam(learning_rate = 0.0001)
compile_metrics = ["accuracy"]
input_dim = (input_images.shape[1], input_images.shape[2], input_images.shape[3])

# remove False once model behaves as expected
if(os.path.exists('model.h5') and False):
    auto_encoder = load_model('model.h5')
else:
    auto_encoder.compile(loss = compile_loss, optimizer= compile_optimizer, metrics = compile_metrics)
    auto_encoder.summary
    auto_encoder_model = auto_encoder.fit(input_images, output_images, validation_data = (test_input, test_output), epochs = 100, callbacks = [stopper, checkpoints], batch_size = 10)
    auto_encoder.save('model.h5')
    
predictions = auto_encoder.predict(input_images)
test_predictions = auto_encoder.predict(test_input)
# np.save(file = "predictions.npy", arr = predictions)
# np.save(file = "test_predictions.npy", arr = test_predictions)
# predictions = np.load(file = "predictions.npy")
# test_img = np.zeros((224, 224, 3))
# test_img[:, :, 0] = x_train[0]
# test_img[:, :, 1:] = predictions[0]
# plt.imshow(test_img)
# plt.show()


def write_imgs(imgs, path, conv):
    if(not os.path.exists(path)):
        os.makedirs(path)
    for i, img in zip(range(len(imgs)), imgs):
        if(conv):
            im = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_LAB2RGB)
        else:
            im = img.astype("uint8")
        im = Image.fromarray(im)
        im.save(path + str(i) + ".JPG")
        
write_imgs(test_predictions, os.getcwd() + "\\testpredictions\\", True)
write_imgs(predictions, os.getcwd() + "\\predictions\\", True)
write_imgs(input_images[:30], os.getcwd() + "\\reference\\", True)
write_imgs(test_input, os.getcwd() + "\\testreference\\", True)
write_imgs(output_images, os.getcwd() + "\\groundtruth\\", True)