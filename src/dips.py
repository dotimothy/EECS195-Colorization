# dips.py Image Processing Functions in opencv #
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang

# Importing Libraries 
import cv2
import os

#Converts the Image to Grayscale and Writes to It
def grayscale(sourcePath,destPath):
	cv2.imwrite(destPath,cv2.cvtColor(cv2.imread(sourcePath),cv2.COLOR_BGR2GRAY))

#Converts a whole folder with images to Grayscale
def grayscaleFolder(source,dest):
	# Loops through directory for images
	for file in os.listdir(source):
		isPicture = file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG")
		if isPicture == True:
			grayscale(source + file, dest + file)	

# Renames an entire Folder of JPG Images to Feed into TF Model
def renameImages(path):
	#Copies into Unique names
	for i, image in enumerate(os.listdir(path)): # enumerate so we can add numbers to file name 
		oldname, ext = os.path.splitext(image)
		os.rename(path + image, path + "photo" + str(i) + ext)

#Prepare a Folder of Images to Train with the Data (path is the image where image dataset is located)
def prepareImageSet(path):
	#Rename all Images in Folders First
	renameImages(path)