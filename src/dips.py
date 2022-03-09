# dips.py Image Processing Functions in opencv #
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang

# Importing Libraries 
import cv2
import os
import time
import numpy as np

#Converts the Image to Grayscale and Writes to It
# DEPRECATED
def grayscale(sourcePath,destPath):
	cv2.imwrite(destPath,cv2.cvtColor(cv2.imread(sourcePath),cv2.COLOR_BGR2GRAY))
	image = cv2.imread(sourcePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.merge([gray,gray,gray])
	cv2.imwrite(destPath,gray)

#Converts a whole folder with images to Grayscale
# DEPRECATED
def grayscaleFolder(source,dest):
	# Loops through directory for images
	for file in os.listdir(source):
		isPicture = file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG")
		if isPicture == True:
			oldname, ext = os.path.splitext(file)
			grayscale(source + file, dest + file)	

# Downscales an Image using openCV
def downScaleImage(sourcePath,destPath):
	image = cv2.imread(sourcePath)
	dim = image.shape
	#ar = dim[0]/dim[1]
	w = 224
	h = 224
	cv2.imwrite(destPath,cv2.resize(image,(w,h)))

# Downscales an Entire Folder
def downScaleFolder(source,dest):
	# Loops through directory for images
	for file in os.listdir(source):
		isPicture = file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG")
		if isPicture == True:
			downScaleImage(source + file, dest + file)	

# Renames an entire Folder of JPG Images to Feed into TF Model
def renameImages(path):
	#Copies into Unique names
	for i, image in enumerate(os.listdir(path)): # enumerate so we can add numbers to file name 
		oldname, ext = os.path.splitext(image)
		if not(os.path.exists(path + "photo"  + str(i) + ext)):
			os.rename(path + image, path + "photo" + str(i) + ext)

# Converts an Image into LAB format as a numpy array
def img2lab(path):
    img = cv2.imread(path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	# normalization from 0 to 100 
    return ((lab / [100, 1, 1]) * [255, 1, 1]) - [0, 128, 128]

# Creating Folder with 313 Classes
# DEPRECATEAD
def classCreate(path):
	if(not os.path.exists(path)):
		os.makedirs(path)
	for i in range(313):
		folder = path + "\\" + str(i) + "\\"
		if(not os.path.exists(folder)): 
			os.makedirs(folder)

# Converts a Folder of Images to L & AB numpy
def setToLab(path):
	np_L = []
	np_ab = []
	for img in os.listdir(path):
		isPicture = img.endswith(".jpg") or img.endswith(".png") or img.endswith(".JPG") or img.endswith(".PNG")
		if isPicture == True:
			lab = img2lab(path + img)
			L = lab[:,:,0]
			ab = lab[:,:,1:2]
			np_L.append(L)
			np_ab.append(ab)
	np.save(path + 'l.npy',np_L)
	np.save(path + 'ab.npy',np_ab)	

# Converts an Image into LAB format, then saves it as a numpy array

#Prepare Folder of Images to Train with the Data (path is the image where image dataset is located)
# Assume Training Path and Validation Path has color and gray folders within them
# Parameters: 
# inputPath: Path to Where All The Original Colorized Images Are
def prepareImageSet(inputPath):
	#Initialize
	imageNumber = len(os.listdir(inputPath))
	prev = time.time()
	current = prev
	
	#Rename all Images in Folders First
	renameImages(inputPath)
	current = time.time()
	print("Rename Images: " + str(current-prev) + " s")
	prev = current
	
	#Downscale all Images in Matlab to 224 by 224
	downScaleFolder(inputPath,inputPath)
	current = time.time()
	print("Downscale Images: " + str(current-prev) + " s")
	prev = current
	
	#Segment all Images in Matlab
	print("------ MATLAB Segmenting Images ------")
	os.system("matlab -nodesktop -nosplash -batch \"segmentFolder '" + inputPath + "' '" + inputPath + "' \"")
	print("------ Matlab Segmenting Complete ------")
	current = time.time()
	print("Segmenting Images: " + str(current-prev) + " s")
	prev = current
	
	
	current = time.time()
	setToLab(inputPath)
	print("Image Set to Numpy Folders: " + str(current-prev) + " s")
	prev = current
		
			






