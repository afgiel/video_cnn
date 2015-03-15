import numpy as np
import os
import pickle


caffe_root = '/root/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe 

print 'setting up alexnet'
caffe.set_mode_gpu()
alexnet = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
	caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

alexnet.transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
alexnet.transformer.set_raw_scale('data', 255)
alexnet.transformer.set_channel_swap('data', (2, 1, 0))

H_SIZE = 227
V_SIZE = 227

DATA_PATH = '/root/data/matrix_data_new/'
class_names = sorted(os.listdir(DATA_PATH)) 

WRITE_PATH = '/root/data/cnn_feats_new/'

print 'extracting'
for class_index in range(len(class_names)):
	class_name = class_names[class_index]
	if 'DS' in class_name:
		continue
	print 'extracting for class %s' % class_name
	write_dir = WRITE_PATH + class_name + '/'
	if not os.path.isdir(write_dir):
		os.mkdir(write_dir) 
	class_dir = DATA_PATH + class_name + '/'
	for pfile in os.listdir(class_dir):
		if not os.path.isfile(os.path.join(WRITE_PATH, class_name, 'rbb-'+ pfile)): # this is a hack 
			if 'DS' in pfile:
				continue	
			f = open(class_dir + pfile, 'r')
			frame_array = pickle.load(f)
			f.close()
			orig_input = frame_array.transpose(0, 3, 1, 2)
			h_diff = int((orig_input.shape[2] - H_SIZE)/2)
			v_diff = int((orig_input.shape[3] - V_SIZE)/2)
			h_offsets = [0, h_diff, 2*h_diff]
			v_offsets = [0, v_diff, 2*v_diff]
			for i in range(len(h_offsets)):
				for j in range(len(v_offsets)):
					h_off = h_offsets[i]
					v_off = v_offsets[j]
					cnn_input = orig_input[:,:,h_off:h_off+H_SIZE,v_off:v_off+V_SIZE] 
					# not flipped
					cnn_out = alexnet.forward(**{alexnet.inputs[0]: cnn_input, 'blobs': ['fc7']})
					cnn_features = cnn_out['fc7'].squeeze(axis=(2,3))	
					write_filename = 'lcr'[i] + 'tmb'[j] + 'f-' + pfile
 					with open(write_dir + write_filename, 'w') as w_pfile:	
						pickle.dump(cnn_features, w_pfile)
					# flipped
					for i in range(cnn_input.shape[1]):
						cnn_input[:,i] = np.fliplr(cnn_input[:,i])  
					cnn_out = alexnet.forward(**{alexnet.inputs[0]: cnn_input, 'blobs': ['fc7']})
					cnn_features = cnn_out['fc7'].squeeze(axis=(2,3))	
					write_filename = 'lcr'[i] + 'tmb'[j] + 'b-' + pfile
					with open(write_dir + write_filename, 'w') as w_pfile:	
						pickle.dump(cnn_features, w_pfile)


