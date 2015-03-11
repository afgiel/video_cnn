# HACKY IMPORTS
from avg.avg import run as avg_run
from avg.avg import get_data as avg_load
#from cnn_over_time.cnn_over_time import run as cnn_over_time_run
#from cnn_over_time.cnn_over_time import get_data as cnn_over_time_load
from rnn.rnn import run as rnn_run
from rnn.rnn import get_data as rnn_load
from rnn.lstm import run as lstm_run
from rnn.lstm import get_data as lstm_load


# FILL IN THESE
num_epochs_choices = [150]
batch_size_choices = [256]
num_hidden_units_choices = [512]
learning_rate_choices = [0.0025]
momentum_choices = [.9]
reg_strength_choices = [0.0025, 0.001, 0.00075]
dropout_choices = [.5]

# CHANGE THE SCRIPT YOU WANT TO RUN 
TEST_LOADER = lstm_load
TEST_RUN = lstm_run
TEST_NAME = 'lstm'

best_val_acc = -1.0
best_params = None

all_params_to_acc = {}

dataset = TEST_LOADER()

for num_epochs in num_epochs_choices:
	for batch_size in batch_size_choices:
		for num_hidden_units in num_hidden_units_choices:
			for learning_rate in learning_rate_choices: 
				for momentum in momentum_choices:
					for reg_strength in reg_strength_choices:
						for dropout in dropout_choices:	
							val_acc = TEST_RUN(dataset,
									num_epochs,
									batch_size,
									num_hidden_units,
									learning_rate,
									momentum,
									reg_strength,
									dropout)
							params = (num_epochs,
								batch_size,
								num_hidden_units,
								learning_rate,
								momentum,
								reg_strength,
								dropout)
							all_params_to_acc[params] = val_acc
							with open(TEST_NAME + '_val.txt', 'a') as output_file:
								output_file.write(str(params) + '\t' + str(val_acc) + '\n')
							if val_acc > best_val_acc:
								best_val_acc = val_acc 
								best_params = params

for params in all_params_to_acc:	
	print params, '\t', all_params_to_acc[params]

with open(TEST_NAME + '_val.txt', 'a') as output_file:
	output_file.write('BEST PARAMS: ' + str(best_params) + '\n')
	output_file.write('BEST ACC: ' + str(best_val_acc) + '\n')
	output_file.write('--------\n')
	


print 'BEST PARAMS'
print best_params
print 'BEST ACC'
print best_val_acc
