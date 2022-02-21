# superpixels.py Performs SLIC algorithm #
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang``
# import the necessary packages
from skimage.segmentation import slic 
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage import data, io, segmentation, color
from skimage.color import rgb2gray
from skimage.future import graph
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import os

# Weighting Functions based on Color Intensities 
def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}
def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

# Grayscale & Segments the Image as a Color
def segmentImage(sourcePath,destPath):

    image_gray = rgb2gray(io.imread(sourcePath))
    image = io.imread(sourcePath)
    
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = np.dstack([image_gray, image_gray, image_gray])
    image_gray = img_as_float(image_gray)
    # load the image and convert it to a floating point data type

    # loop over the number of segments
    #for numSegments in (100,200,300):
    numSegments = 10000
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image_gray, n_segments = numSegments, sigma = 5)
    g = graph.rag_mean_color(image,segments)
    labels2 = graph.merge_hierarchical(segments, g, thresh=35, rag_copy=False,
                                    in_place_merge=True,
                                    merge_func=merge_mean_color,
                                    weight_func=_weight_mean_color)
    out = color.label2rgb(labels2, image, kind='avg', bg_label=0)

    
    #out = segmentation.mark_boundaries(out, labels2, color=(0,0,0))
    # saves segmented image
    # fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image,segments,color=(0,0,0)))

    # Saving Image
    io.imsave(destPath,img_as_ubyte(out))

    #Prints Every Single Segment
    # for (i, segVal) in enumerate(np.unique(segments)):
    #     print("[x] inspecting segment %d" % (i))
    #     mask = np.zeros(image.shape[:2], dtype = "uint8")
    #     mask[segments == segVal] = 255
    #     # show the masked region
    #     cv2.imshow("Mask", mask)
    #     cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
    #     cv2.waitKey(1)

    #plt.axis("off")
    # show the plots
    #plt.show()

# Segments
def segmentFolder(source,dest):
	# Loops through directory for images
	for file in os.listdir(source):
		isPicture = file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG")
		if isPicture == True:
			segmentImage(source + file, dest + file)	

