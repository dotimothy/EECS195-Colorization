# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "../images/flowers/IMG_9018 (1).JPG")
args = vars(ap.parse_args())
image = img_as_float(io.imread(args["image"]))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
m = np.dstack([gray, gray, gray])
# load the image and convert it to a floating point data type

# loop over the number of segments
#for numSegments in (100,200,300):
numSegments = 500
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(m, n_segments = numSegments, sigma = 5)
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(m,segments,color=(0,0,0)))
plt.axis("off")

# show the plots
plt.show()