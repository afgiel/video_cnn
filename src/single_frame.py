# our imports 
import load_data

# scientific imports
import numpy as np
import lasagne
import theano 
import theano.tensor as T

NUM_EPOCHS = 10
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 256
LEARNING RATE = 0.01
MOMENTUM = 0.9

def build_model(input_dim, output_dim, 
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):

    l_in = lasagne.layers.InputLayer(
          shape = (batch_size, input_dim),
          )
    l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=32,
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(),
            )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))
    l_hidden1 = lasagne.layers.DenseLayer(
            l_pool2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Uniform(),
            )
    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    l_out = lasagne.layers.DenseLayer(
            l_hidden1_dropout,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Uniform(),
            )

    return l_out

def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,
                          momentum=MOMENTUM):
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(
      batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.multinomial_nll)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
      loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
      [batch_index], loss_train,
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
    num_batches_test = dataset['num_examples_test'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

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
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            }

def main(num_epochs=NUM_EPOCHS):
  dataset = load_data.load_data()
  output_layer = build_model(
    input_dim = dataset['input_dim'],
    output_dim = dataset['output_dim'],
    )
  iter_funcs = create_iter_functions(dataset, output_layer)

  print 'TRAINING'
  for epoch in train(iter_funcs, dataset):
    print("Epoch %d of %d" % (epoch['number'], num_epochs))
    print("\ttraining loss:\t\t%.6f" % epoch['train_loss'])
    print("\tvalidation loss:\t\t%.6f" % epoch['valid_loss'])
    print("\tvalidation accuracy:\t\t%.2f %%" %
            (epoch['valid_accuracy'] * 100))

    if epoch['number'] >= num_epochs:
      break

  return output_layer


if __name__ == '__main__':
    main()
