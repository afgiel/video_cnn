import os
import pickle
from PIL import Image

DATA_PATH = '../../../data/matrix_data_rnn/'
class_names = sorted(os.listdir(DATA_PATH)) 

WRITE_PATH = '../../../data/single_frames/'

print 'extracting'
for class_index in range(len(class_names)):
	class_name = class_names[class_index]
	print 'extracting single frames for class %s' % class_name
	class_dir = DATA_PATH + class_name + '/'
	for pfile in os.listdir(class_dir):
		if not os.path.isfile(os.path.join(WRITE_PATH, class_name, pfile)):
			if 'DS' in pfile:
				continue	
			f = open(class_dir + pfile, 'r')
			frame_array = pickle.load(f)
			one_frame = frame_array[-1]
			one_frame[:,:,[0,1,2]] = one_frame[:,:,[2,1,0]]
			im = Image.fromarray(one_frame)
			im_file = pfile[:-6] + 'png'
			f.close()
			write_dir = WRITE_PATH + class_name + '/'
			if not os.path.isdir(write_dir):
				os.mkdir(write_dir) 
			im.save(write_dir + im_file)	


