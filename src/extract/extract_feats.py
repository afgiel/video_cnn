import numpy as np
import os
import pickle

caffe_root = '../../../caffe/'
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

DATA_PATH = '../../../data/matrix_data_rnn/'
class_names = sorted(os.listdir(DATA_PATH)) 

WRITE_PATH = '../../../data/cnn_feats/'

print 'extracting'
for class_index in range(len(class_names)):
	class_name = class_names[class_index]
	print 'extracting for class %s' % class_name
	class_dir = DATA_PATH + class_name + '/'
	for pfile in os.listdir(class_dir):
		if not os.path.isfile(os.path.join(WRITE_PATH, class_name, pfile)):
			if 'DS' in pfile:
				continue	
			f = open(class_dir + pfile, 'r')
			frame_array = pickle.load(f)
			f.close()
			cnn_input = frame_array.transpose(0, 3, 1, 2)
			cnn_out = alexnet.forward(**{alexnet.inputs[0]: cnn_input})
			cnn_features = cnn_out[alexnet.outputs[0]].squeeze(axis=(2,3))
			write_dir = WRITE_PATH + class_name + '/'
			if not os.path.isdir(write_dir):
				os.mkdir(write_dir) 
			with open(write_dir + pfile, 'w') as w_pfile:	
				pickle.dump(cnn_features, w_pfile)


#print 'extracting'
#for batch, class_name, filename in images_and_class:
#	print 'extracting for %s : filename' % (class_name, filename)
#	batch_out = alexnet.forward(**{alexnet.inputs[0]: batch})
#	batch_features = batch_out[alexnet.outputs[0]].squeeze(axis=(2,3))
#	class_dir = WRITE_PATH + class_name + '/'
#	if not os.path.isdir(class_dir):
#		os.mkdir(class_dir) 
#	with open(class_dir + filename, 'w') as pfile:	
#		batch_features_pickle = pickle.dump(batch_features, pfile)
#print 'done extracting'
