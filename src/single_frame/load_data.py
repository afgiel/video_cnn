import itertools
import pickle
import os
import sys


# scientific imports 
import numpy as np
import theano
import theano.tensor as T
import lasagne



def _load_data():
    data_dir = '../data/matrix_data/'
    classes = os.listdir(data_dir)[1:]
    classes[:3]
    num_classes = len(classes)
    data = [([],[]),([],[]),([],[])]
    # single frame for
    for class_val in range(num_classes):
        print 'loading class: ', class_val
        clips = os.listdir(data_dir + classes[class_val])[1:]
        # limit the list size for now 
        clips = clips[:1]
        split_size = len(clips)/10
        train_size = len(clips) - 2*split_size
        val_ind = train_size - 1 + split_size
        arrays = []
        for clip in clips:
            f = open(data_dir + classes[class_val] + '/' + clip)
            print data_dir + classes[class_val] + '/' + clip
            frames = pickle.load(f)
            print frames.shape
            arrays.append(frames)
            f.close()
        
        data[0][0].extend(arrays[:train_size])
        data[0][1].extend([class_val]*train_size)
        data[1][0].extend(arrays[train_size-1:val_ind])
        data[1][1].extend([class_val]*split_size)
        data[2][0].extend(arrays[val_ind-1:])
        data[2][1].extend([class_val]*split_size)
        return data
            

def load_data():
    #data = _load_data()
    print '\tunpickling'
    data = pickle.load(open('../data/small_data.pickle'))
    print '\tnumpying' 
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # change shape of X, y for single frame usage
    X_train, y_train = reshape_for_single_frame(X_train, y_train)
    X_valid, y_valid = reshape_for_single_frame(X_valid, y_valid)
    X_test, y_test = reshape_for_single_frame(X_test, y_test)

    print '\tcreating theano shared vars'
    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape,
        output_dim=101,
        )

# expects X of size (N, F, C, H, W)
# expects y of size (N)
# returns X of size (N * F, C, H, W)
# returns y of size (N * F)
def reshape_for_single_frame(X, y):
  X = X.transpose(0, 1, 4, 2, 3) 
  X_shape = X.shape
  y_shape = y.shape
  new_X = X.reshape((X_shape[0] * X_shape[1], X_shape[2], X_shape[3], X_shape[4]))
  new_y = np.zeros((y_shape[0] * X_shape[1]))
  for i in xrange(y_shape[0]):
    j = i * X_shape[1]
    new_y[j:j+X_shape[1]] = y[i]
  return new_X, new_y

