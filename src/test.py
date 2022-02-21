# Test Function to Grayscale a Test Folder ("gray" from "color")
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang
import dips
import superpixels as sp
import time
import os
	
# Directories
sourceName = os.getcwd() + "\\color\\"
altName = os.getcwd() + "\\color1\\"
destName = os.getcwd() + "\\down\\"
testSource = os.getcwd() + "\\red-rose.jpg"
testDest = os.getcwd() + "\\seg\\red-rose.jpg"
segName = os.getcwd() + "\\seg\\"

# Start test
start = time.time()
dips.downScaleFolder(sourceName,destName)
#sp.segmentFolder(destName,segName)
duration = time.time() - start
print(str(duration) + " seconds")