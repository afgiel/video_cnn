import numpy as np
import os
import pickle
import itertools
import time
import random

import matplotlib.pyplot as plt

from tsne import tsne as tsne
from tsne import pca

DATA_PATH = '/root/data/cnn_feats/'
NUM_CLASSES = 10

def get_data():
	#get data 
	class_names = sorted(os.listdir(DATA_PATH))[:NUM_CLASSES] 

	print '\tgetting data'
	image_feats = []
	labels = []
	for class_index in range(len(class_names)):
		class_name = class_names[class_index]
		print '\tgetting data for class %s' % class_name
		class_dir = DATA_PATH + class_name + '/'
		for pfile in os.listdir(class_dir):
			f = open(class_dir + pfile, 'r')
			features = pickle.load(f)
			f.close()
			image_feats.append(features)
			labels.append(class_index)

	# avg each input 
	print '\taveraging each'
	inputs = []
	for image_feat in image_feats:
		inputs.append(np.average(image_feat, axis=0))	
	
	
	print '\tnumpying' 
	X = np.array(inputs, dtype=np.float64)
 	y = np.array(labels)
	
	return X, y

def plot_tsne(tsne_X, labels):
	print tsne_X[0:5]
	color_choices = ['red', 'coral', 'orange', 'yellow', 'green', 'aqua', 'blue', 'violet', 'black', 'brown'] 
	x = []
	y = []
	colors = []
	for i in range(tsne_X.shape[0]):
		dim1, dim2 = tsne_X[i]
		x.append(dim1)
		y.append(dim2) 
		colors.append(color_choices[labels[i]])	
	plt.scatter(x, y, c=colors)
	plt.savefig('avg_cnn_tsne')

def main():	
	print 'LOADING DATA'
	X, y = get_data()	
	print 'TSNE-ing'
	#tsne_X = tsne(X, no_dims=2, initial_dims=4096) 		
	tsne_X = pca(X, no_dims=2)
	plot_tsne(tsne_X, y)	
	
if __name__ == '__main__':
    main()
