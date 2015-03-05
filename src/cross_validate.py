# HACKY IMPORTS
from avg.avg import run as avg_run
from avg.avg import get_data as avg_load
#from cnn_over_time.cnn_over_time import run as cnn_over_time_run
#from cnn_over_time.cnn_over_time import get_data as cnn_over_time_load


# FILL IN THESE
num_epochs_choices = [125, 150, 175]
batch_size_choices = [256]
num_hidden_units_choices = [512]
learning_rate_choices = [0.00025, 0.0005, 0.00075]
momentum_choices = [.9]
reg_strength_choices = [.0001, 0.00001, 0.]
dropout_choices = [.6, .7]

# CHANGE THE SCRIPT YOU WANT TO RUN 
TEST_LOADER = avg_load
TEST_RUN = avg_run

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
							if val_acc > best_val_acc:
								best_val_acc = val_acc 
								best_params = params



for params in all_params_to_acc:
	print params, '\t', all_params_to_acc[params]

print 'BEST PARAMS'
print best_params
print 'BEST ACC'
print best_val_acc
