# superpixels.py Performs SLIC algorithm #
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang``
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

# Segments the Image
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "")
args = vars(ap.parse_args())

# image = rgb2gray(io.imread(args["image"]))
image = img_as_float(io.imread(args["image"]))


#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = np.dstack([image, image, image])
# load the image and convert it to a floating point data type

# loop over the number of segments
#for numSegments in (100,200,300):
numSegments = 1000
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, n_segments = numSegments, sigma = 5)
print(segments[0])
# show the output of SLIC
#fig = plt.figure("Superpixels -- %d segments" % (numSegments))
#ax = fig.add_subplot(1, 1, 1)
#ax.imshow(mark_boundaries(image,segments,color=(0,0,0)))
#io.imsave("segnment.jpg",img_as_uint(mark_boundaries(image,segments,color=(0,0,0))))

#Prints Every Single Segment
for (i, segVal) in enumerate(np.unique(segments)):
    print("[x] inspecting segment %d" % (i))
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    mask[segments == segVal] = 255
    # show the masked region
    cv2.imshow("Mask", mask)
    cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
    cv2.waitKey(1)

#plt.axis("off")
# show the plots
#plt.show()