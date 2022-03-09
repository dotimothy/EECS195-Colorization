from PIL import Image
import numpy as np
from skimage import color
from cv2 import imshow
from skimage.segmentation import slic 
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import data, io, segmentation, color
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.future import graph
import matplotlib.pyplot as plt
import argparse
import cv2
from torch import threshold

#######Variables 
maxColorPrediction = maxRedPrediction = maxGreenPrediction = maxBluePrediction = 0
maxRedTotal = maxGreenTotal = maxBlueTotal = 1
maxRedThreshold = maxGreenThreshold = maxBlueThreshold = 0
ThresholdMet = GRAYCount = CVGrayCount = 0
grayThresh = 20
Threshold = 50
def resize_img(img):
	return resize(img,(256,256))
def maxChannel(photo):
    if photo[0] > photo[1] and photo[0] > photo[2]:
        return 0
    if photo[1] > photo[0] and photo[1] > photo[2]:
        return 1    
    else:
        return 2
def thresholdmet(threshold,original,predicted):
    i = maxChannel(original)
    if (predicted[0] - original[0]) < threshold and (predicted[0] - original[0]) > -threshold:
        if (predicted[1] - original[1]) < threshold and (predicted[1] - original[1]) > -threshold:
            if (predicted[2] - original[2]) < threshold and (predicted[2] - original[2]) > -threshold:    
                return 1
    return 0
def CorrectMax(value):
    if(value == 0):
        maxRedPrediction = maxRedPrediction + 1
    if(value == 1):
        maxGreenPrediction = maxGreenPrediction + 1
    if(value == 2):
        maxBluePrediction = maxBluePrediction + 1
def grayScaleMet(grayThresh, original,predicted):
    averageOG = (original[0] + original[1] + original[2])/3
    predictedOG = (predicted[0]+ predicted[1]+ predicted[2])/3
    if(predictedOG > averageOG-grayThresh) and (predictedOG< averageOG+grayThresh):
        return 1
    return 0 
def OPENCVGrayScaleFromula(grayThresh,original,predicted):
    averageOG = (original[0]*.30 + original[1]*.59 + original[2]*.11)
    predictedOG = (predicted[0]*.30+ predicted[1]*.59+ predicted[2]*.11)
    if(predictedOG > averageOG-grayThresh) and (predictedOG< averageOG+grayThresh):
        return 1
    return 0 

def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}
def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

OriginalImageName = r'\white_rose.jpg' #Name of input image (Needs to be 256x256)  
PredictedImageName = r'\PredictedWhiteRose.png' #Name of the Predicted Image             
ColorImage = r"C:\Users\Matthew Prata\Desktop\School\Winter 2022\EECS 195\Project\SquareImages" + OriginalImageName
PredictedImage = r"C:\Users\Matthew Prata\Desktop\School\Winter 2022\EECS 195\Project\PredictedImages" + PredictedImageName
#image_color = Image.open(ColorImage)
image_color = io.imread(ColorImage)

#image_color = resize_img(image_color)
image_predicted = io.imread(PredictedImage)
IMAGETEST = Image.open(PredictedImage)
IMAGETEST = np.asarray(IMAGETEST)
print(type(IMAGETEST))
# io.imshow(image_color)
# io.show()

image_predicted=image_predicted[:,:,:3]
image = rgb2gray(io.imread(ColorImage))
predicted_gray = rgb2gray(image_predicted)
image = img_as_float(image)
img = cv2.imread(ColorImage,0)

image = np.dstack([image, image, image])
predicted_gray = np.dstack([predicted_gray, predicted_gray,predicted_gray])

numSegments = 10000

segments = slic(image, n_segments = numSegments, sigma = 5)
segments2 = slic(predicted_gray, n_segments = numSegments, sigma = 5) 
g = graph.rag_mean_color(image_color,segments)
g1 = graph.rag_mean_color(image_predicted,segments2)

labels2 = graph.merge_hierarchical(segments, g, thresh=35, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
labels3 = graph.merge_hierarchical(segments, g1, thresh=35, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

out = color.label2rgb(labels2, image_color, kind='avg', bg_label=0)
out2 = color.label2rgb(labels3, image_predicted, kind='avg', bg_label=0)
#Initializing Variables 



for x in range(256):
    for y in range(256):
        out2RGB = out2[x][y]
        outRGB = out[x][y]
        Channel2 = maxChannel(out2RGB)
        Channel1 = maxChannel(outRGB)
        if(Channel1 == 0):
            maxRedTotal = maxRedTotal + 1
        if(Channel1 == 1):
            maxGreenTotal = maxGreenTotal + 1
        if(Channel1 == 2):
            maxBlueTotal = maxBlueTotal + 1
        if Channel2 == Channel1:
            maxColorPrediction = maxColorPrediction + 1      
            if(Channel2 == 0):
                maxRedPrediction = maxRedPrediction + 1
            if(Channel2 == 1):
                maxGreenPrediction = maxGreenPrediction + 1
            if(Channel2 == 2):
                maxBluePrediction = maxBluePrediction + 1
        
        Check = thresholdmet(Threshold,outRGB,out2RGB)
        if(Check == 1):
            ThresholdMet = ThresholdMet + 1
        GrayScaleCheck = grayScaleMet(grayThresh,outRGB,out2RGB)
        if(GrayScaleCheck == 1):
            GRAYCount = GRAYCount + 1
        CVGrayScaleCheck = OPENCVGrayScaleFromula(grayThresh,outRGB,out2RGB)
        if(CVGrayScaleCheck == 1):
            CVGrayCount = CVGrayCount+1

        
def printGrade():
    print("\n--------------------------------------------------------")
    print(f"Max Color Probabilty: {maxColorPrediction/(256*256)}")   
    
    print(f"Max Red Probabilty: {maxRedPrediction/maxRedTotal} -> {maxRedPrediction}/{maxRedTotal}")
    print(f"Max Green Probabilty: {maxGreenPrediction/maxGreenTotal} -> {maxGreenPrediction}/{maxGreenTotal}")
    print(f"Max Blue Probabilty: {maxBluePrediction/maxBlueTotal} -> {maxBluePrediction}/{maxBlueTotal}") 
    #print(f"Color Threshold Met: {ThresholdMet/(256*256)} Threshold: {Threshold}")
    print(f"GrayScale Met with Threshold: {GRAYCount/(256*256)} Threshold: {grayThresh} ")
    print(f"GrayScale OPENCV comparision: {CVGrayCount/(256*256)} Threshold: {grayThresh}")
    #print(f"Max Threshold Probabilty R: {maxRedThreshold/maxRedTotal} G: {maxGreenThreshold/maxGreenTotal} B: {maxBlueThreshold/maxBlueTotal} ")
    print("--------------------------------------------------------")         
printGrade()   
print(len(np.unique(out2,axis = 0)))
print("----------------------------")
print(len(out2))
#for (i, segVal) in enumerate(np.unique(out2)):
    #print("[x] inspecting segment %d" % (i))
    
    # mask = np.zeros(image.shape[:2], dtype = "uint8")
    # mask[segments == segVal] = 255
    #print(f"Boundaries: {mark_boundaries(image,segments,color=(0,0,0))}\n")
    #print(f"Mask Segments: {segments}")
    # show the masked region
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
    #cv2.waitKey(1)

#Displaying Images#######################                                                                             
plt.figure(figsize = (12,8))            #
plt.subplot(2,2,1)                      #
plt.imshow(out)                         #
plt.title('Original Photo')             #
plt.axis('off')                         #
                                        #
plt.subplot(2,2,2)                      #
plt.imshow(out2)                        #
plt.title('Predicted Photo')            #
plt.axis('off')                         #
                                        # 
plt.subplot(2,2,3)                      #
plt.imshow(image_color)                 #
plt.title('Input')                      #
plt.axis('off')                         #
                                        #
plt.subplot(2,2,4)                      #
plt.imshow(image_predicted)             #
plt.title('Predicted Image')            #
plt.axis('off')                         #
plt.show()                              #
#########################################



out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

 

