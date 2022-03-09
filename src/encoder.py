from sklearn.cluster import KMeans
import numpy as np
import cv2
import os
import dips

#finds the bin centers of histograms of images
def get_bin_centers(path):
	bin_centers = []
	imgs = []
	for file in os.listdir(path):
		img = dips.img2lab(path + file)
		img = img[:, :, 1]  + img[:, :, 2]
		imgs.append(img)
		bin_edges = np.histogram_bin_edges(img, bins = 10)
		bin_centers.append(bin_edges[:-1] + np.diff(bin_edges)/2)
	np.save(file = "bin_centers.npy", arr = bin_centers)
	return bin_centers, imgs

#creates the folders based off of the labels found by KMeans
def classCreate(path, source):
	kmeans = KMeans(n_clusters = 238)
	labels = kmeans.fit_predict(np.array(get_bin_centers(source)[0]))
	imgs = get_bin_centers(source)[1]
	if(not os.path.exists(path)):
		os.makedirs(path)
	for i in range(len(labels)):
		folder = path + "\\" + str(labels[i]) + "\\"
		if(not os.path.exists(folder)): 
			os.makedirs(folder)
		write_path = os.getcwd() + "\\" + path + "\\" + str(labels[i]) +  "\\" + str(i) +".JPG"
		cv2.imwrite(write_path, imgs[i])