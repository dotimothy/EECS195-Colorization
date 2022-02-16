# Test Function to Grayscale a Test Folder ("gray" from "color")
# Authors: Timothy Do, Matthew Prata, Jorge Radge, Alex Wang
import dips
import time
import os
	
sourceName = os.getcwd() + "\\color\\"
destName = os.getcwd() + "\\gray\\"
dips.renameImages(sourceName)
dips.grayscaleFolder(sourceName,destName)
