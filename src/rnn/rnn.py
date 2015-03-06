import numpy as np
import os
import pickle
import itertools
import time
import random

import theano
from theano import tensor as T
import lasagne
from recurrent import RecurrentLayer, RecurrentSoftmaxLayer

NUM_EPOCHS = 200
BATCH_SIZE = 256
NUM_HIDDEN_UNITS = 512
LEARNING_RATE = 0.00005
MOMENTUM = 0.9
REG_STRENGTH = 0.0001
DROPOUT = 0.6

SEED = .42

DATA_PATH = '/root/data/cnn_feats/'


def get_data():
	#get data 
	class_names = sorted(os.listdir(DATA_PATH)) 

	print '\tgetting data'
	inputs = []
	labels = []
	for class_index in range(len(class_names)):
		class_name = class_names[class_index]
		print '\tgetting data for class %s' % class_name
		class_dir = DATA_PATH + class_name + '/'
		for pfile in os.listdir(class_dir):
			f = open(class_dir + pfile, 'r')
			features = pickle.load(f)
			f.close()
			inputs.append(features)
			labels.append(class_index)

	random.shuffle(inputs, lambda: SEED)
	random.shuffle(labels, lambda: SEED)
	assert len(inputs) == len(labels)
				
	split_size = len(inputs)/10
        train_size = len(inputs) - 2*split_size
        val_ind = train_size - 1 + split_size
	
	X_train = inputs[:train_size]	
	y_train = labels[:train_size]
	X_valid = inputs[train_size-1:val_ind]
	y_valid = labels[train_size-1:val_ind]
	X_test = inputs[val_ind-1:]
	y_test = labels[val_ind-1:]

	print '\tnumpying' 
	X_train = np.array(X_train)
 	y_train = np.array(y_train)
 	X_valid = np.array(X_valid)
 	y_valid = np.array(y_valid)
 	X_test = np.array(X_test)
 	y_test = np.array(y_test)

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
		output_dim=len(class_names),
		)	

def build_model(input_dim, output_dim, 
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
	l_in = lasagne.layers.InputLayer(
          shape=(batch_size, input_dim[1], input_dim[2]),
          )
	l_rec1 = RecurrentLayer(
            l_in,
            num_units=num_hidden_units
            ) 
	l_out = RecurrentSoftmaxLayer(
	    l_rec1,
	    num_units=output_dim
	    )
	return l_out

def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.tensor3,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,
                          momentum=MOMENTUM,
                          reg_strength=REG_STRENGTH):
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(
      batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.multinomial_nll)

    reg = lasagne.regularization.l2(output_layer) 
    loss_train = objective.get_loss(X_batch, target=y_batch) + REG_STRENGTH*reg
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.rmsprop(
      loss_train, all_params, learning_rate, momentum)
    

    iter_train = theano.function(
      [batch_index], [loss_train, accuracy],
      updates=updates,
      givens={
        X_batch: dataset['X_train'][batch_slice],
        y_batch: dataset['y_train'][batch_slice],
        },
      )

    iter_valid = theano.function(
      [batch_index], [loss_eval, accuracy],
      givens={
        X_batch: dataset['X_valid'][batch_slice],
        y_batch: dataset['y_valid'][batch_slice],
        },
      )

    iter_test = theano.function(
      [batch_index], [loss_eval, accuracy],
      givens={
        X_batch: dataset['X_test'][batch_slice],
        y_batch: dataset['y_test'][batch_slice],
        },
      )

    return dict(
      train=iter_train,
      valid=iter_valid,
      test=iter_test,
      )

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
	batch_train_accuracies = []
        for b in range(num_batches_train):
            #print '\tbatch %d of %d' % (b, num_batches_train)
            #tick = time.time()
            batch_train_loss, batch_train_accuracy = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)
	    batch_train_accuracies.append(batch_train_accuracy)
            #toc = time.time()
            #print '\t\t loss: %f' % (batch_train_loss)
            #print '\t\t took %f' % (toc - tick)

        avg_train_loss = np.mean(batch_train_losses)
	avg_train_accuracy = np.mean(batch_train_accuracies)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
	    'train_accuracy': avg_train_accuracy,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy
            }

def test(iter_funcs, dataset, batch_size=BATCH_SIZE):
	num_batches_test = dataset['num_examples_test'] // batch_size
	batch_accuracies = []
	for b in range(num_batches_test):
		batch_loss, batch_accuracy = iter_funcs['test'](b)
		batch_accuracies.append(batch_accuracy)
	avg_test_accuracy = np.mean(batch_accuracies)	
	return avg_test_accuracy

def run(dataset,
	num_epochs=NUM_EPOCHS,
	batch_size=BATCH_SIZE,
	num_hidden_units=NUM_HIDDEN_UNITS,
	learning_rate=LEARNING_RATE,
	momentum=MOMENTUM,
	reg_strength=REG_STRENGTH,
	dropout=DROPOUT,
	TESTING = False
	):
	# assign global vars
	global NUM_EPOCHS
	global BATCH_SIZE
	global NUM_HIDDEN_UNITS 
	global LEARNING_RATE 
	global MOMENTUM
	global REG_STRENGTH
	global DROPOUT
	NUM_EPOCHS = num_epochs
	BATCH_SIZE = batch_size
	NUM_HIDDEN_UNITS = num_hidden_units 
	LEARNING_RATE = learning_rate 
	MOMENTUM = momentum
	REG_STRENGTH = reg_strength
	DROPOUT = dropout

	to_return = None

	print 'BUILDING MODEL'
	output_layer = build_model(
	    input_dim = dataset['input_dim'],
	    output_dim = dataset['output_dim'],
	    )
	print 'CREATING ITER FUNCS'
	iter_funcs = create_iter_functions(dataset, output_layer)

	print 'TRAINING'
	for epoch in train(iter_funcs, dataset):
		print("Epoch %d of %d" % (epoch['number'], num_epochs))
		print("\ttraining loss:\t\t%.6f" % epoch['train_loss'])
		print("\ttraining accuracy:\t\t%.2f %%" % (epoch['train_accuracy'] * 100))
		print("\tvalidation loss:\t\t%.6f" % epoch['valid_loss'])
		validation_acc = (epoch['valid_accuracy'] * 100)
		print("\tvalidation accuracy:\t\t%.2f %%" % (validation_acc))

		to_return = validation_acc

		if epoch['number'] >= num_epochs:
			break
	if TESTING:	
		print 'TESTING'
		test_acc = test(iter_funcs, dataset)	
		print 'test accuracy: \t\t%.2f' % test_acc 
		to_return = test_acc

	return to_return


def main(num_epochs=NUM_EPOCHS,
	batch_size=BATCH_SIZE,
	num_hidden_units=NUM_HIDDEN_UNITS,
	learning_rate=LEARNING_RATE,
	momentum=MOMENTUM,
	reg_strength=REG_STRENGTH,
	dropout=DROPOUT,
	TESTING = False
	):
	
	print 'LOADING DATA'
	dataset = get_data()	
	return run(dataset,
		num_epochs,
		batch_size,
		num_hidden_units,
		learning_rate,
		momentum,
		reg_strength,
		dropout)
	
if __name__ == '__main__':
    main()
