import os, sys, random
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

import caffe
caffe.set_device(0)


DATA_PATH = '/root/data/single_frames/'

classes = os.listdir(DATA_PATH)

class_to_idx = {a: i for i, a in enumerate(classes)}

def make_caffe_input_file(path):
  train_out = open(path + 'train.txt', 'w')
  val_out = open(path + 'val.txt', 'w')
  test_out = open(path + 'test.txt', 'w')
  for c in classes:
	idx = class_to_idx[c]
	images = os.listdir(DATA_PATH + c)
	split_size = len(images)/10
	for img_idx in range(len(images)):
		image_loc = DATA_PATH + c + '/' + images[img_idx]
		if img_idx < split_size: # val
			val_out.write('%s %d\n' % (image_loc, idx))
		elif img_idx < 2*split_size: # test
			test_out.write('%s %d\n' % (image_loc, idx)) 
		else: # train
			train_out.write('%s %d\n' % (image_loc, idx))  

make_caffe_input_file('/root/data/caffe_single_frames/train_val_test/')
