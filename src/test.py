# Test Function to Grayscale a Test Folder ("gray" from "color")
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang
import dips
import time
import os
	
# Directories
sourceName = os.getcwd() + "\\color\\"
altName = os.getcwd() + "\\seg1\\"
destName = os.getcwd() + "\\gray\\"
altDestName = os.getcwd() + "\\gray1\\"
testSource = os.getcwd() + "\\red-rose.jpg"
testDest = os.getcwd() + "\\seg\\red-rose.jpg"
segName = os.getcwd() + "\\seg\\"

input  = os.getcwd() + "\\color1\\"
gray = os.getcwd() + "\\gray1\\"
train = os.getcwd() + "\\train\\"
val = os.getcwd() + "\\val\\"

# Start test
start = time.time()
#dips.prepareImageSet(input,gray,train,val,0.5)
dips.grayscaleFolder(sourceName,destName)
duration = time.time() - start
print("Total Time: " +  str(duration) + " seconds")