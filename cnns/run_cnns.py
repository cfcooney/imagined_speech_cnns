"""
Name: Ciaran Cooney
Date: 12/01/2019
Description: Training, validation and testing of three CNNs using a 
nested cross-vlaidation scheme for hyper-parameter selection. Optimized
hyper-parameters are used to train a new model
"""

import numpy as np
import pandas as pd 
import logging  
import time
import sys 
from utils import load_subject_eeg, eeg_to_3d, balanced_subsample, format_data, current_loss

from sklearn.model_selection import train_test_split, StratifiedKFold

#####import network architectures#####
#from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from eegnet import EEGNetv4
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.functions import square, safe_log
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or, And
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint 
from experiment_sans_test import Experiment 
from experiment import Experiment as op_exp # experiemnt for saving optimized models
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor 
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import set_random_seeds, np_to_var

from torch.nn.functional import elu, relu6, leaky_relu, relu, rrelu
import torch 
import torch.nn.functional as F 
from torch import optim

from tensorflow.keras.utils import normalize
torch.backends.cudnn.deterministic = True

log = logging.getLogger(__name__)

def network_model(subject_id, model_type, data_type, cropped, cuda, parameters, hyp_params):
	best_params = dict() # dictionary to store hyper-parameter values

	#####Parameter passed to funciton#####
	max_epochs  = parameters['max_epochs']
	max_increase_epochs = parameters['max_increase_epochs']
	batch_size = parameters['batch_size']

	#####Constant Parameters#####
	best_loss = 100.0 # instatiate starting point for loss
	iterator = BalancedBatchSizeIterator(batch_size=batch_size)
	stop_criterion = Or([MaxEpochs(max_epochs),
						 NoDecrease('valid_misclass', max_increase_epochs)])
	monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
	model_constraint = MaxNormDefaultConstraint()
	epoch = 4096

	#####Collect and format data#####
	if data_type == 'words':
		data, labels = format_data(data_type, subject_id, epoch)
		data = data[:,:,768:1280] # within-trial window selected for classification
	elif data_type == 'vowels':
		data, labels = format_data(data_type, subject_id, epoch)
		data = data[:,:,512:1024]
	elif data_type == 'all_classes':
		data, labels = format_data(data_type, subject_id, epoch)
		data = data[:,:,768:1280]
	
	x = lambda a: a * 1e6 # improves numerical stability
	data = x(data)
	
	data = normalize(data)
	data, labels = balanced_subsample(data, labels) # downsampling the data to ensure equal classes
	data, _, labels, _ = train_test_split(data, labels, test_size=0, random_state=42) # redundant shuffle of data/labels

	#####model inputs#####
	unique, counts = np.unique(labels, return_counts=True)
	n_classes = len(unique)
	n_chans   = int(data.shape[1])
	input_time_length = data.shape[2]

	#####k-fold nested corss-validation#####
	num_folds = 4
	skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
	out_fold_num = 0 # outer-fold number
	
	cv_scores = []
	#####Outer=Fold#####
	for inner_ind, outer_index in skf.split(data, labels):
		inner_fold, outer_fold     = data[inner_ind], data[outer_index]
		inner_labels, outer_labels = labels[inner_ind], labels[outer_index]
		out_fold_num += 1
		 # list for storing cross-validated scores
		loss_with_params = dict()# for storing param values and losses
		in_fold_num = 0 # inner-fold number
		
		#####Inner-Fold#####
		for train_idx, valid_idx in skf.split(inner_fold, inner_labels):
			X_Train, X_val = inner_fold[train_idx], inner_fold[valid_idx]
			y_train, y_val = inner_labels[train_idx], inner_labels[valid_idx]
			in_fold_num += 1
			train_set = SignalAndTarget(X_Train, y_train)
			valid_set = SignalAndTarget(X_val, y_val)
			loss_with_params[f"Fold_{in_fold_num}"] = dict()
			
			####Nested cross-validation#####
			for drop_prob in hyp_params['drop_prob']:
				for loss_function in hyp_params['loss']:
					for i in range(len(hyp_params['lr_adam'])):
						model = None # ensure no duplication of models
						# model, learning-rate and optimizer setup according to model_type
						if model_type == 'shallow':
							model =  ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes, input_time_length=input_time_length,
										 n_filters_time=80, filter_time_length=40, n_filters_spat=80, 
										 pool_time_length=75, pool_time_stride=25, final_conv_length='auto',
										 conv_nonlin=square, pool_mode='max', pool_nonlin=safe_log, 
										 split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
										 drop_prob=drop_prob).create_network()
							lr = hyp_params['lr_ada'][i]
							optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, weight_decay=0.1, eps=1e-8)
						elif model_type == 'deep':
							model = Deep4Net(in_chans=n_chans, n_classes=n_classes, input_time_length=input_time_length,
										 final_conv_length='auto', n_filters_time=20, n_filters_spat=20, filter_time_length=10,
										 pool_time_length=3, pool_time_stride=3, n_filters_2=50, filter_length_2=15,
										 n_filters_3=100, filter_length_3=15, n_filters_4=400, filter_length_4=10,
										 first_nonlin=leaky_relu, first_pool_mode='max', first_pool_nonlin=safe_log, later_nonlin=leaky_relu,
										 later_pool_mode='max', later_pool_nonlin=safe_log, drop_prob=drop_prob, 
										 double_time_convs=False, split_first_layer=False, batch_norm=True, batch_norm_alpha=0.1,
										 stride_before_pool=False).create_network() #filter_length_4 changed from 15 to 10
							lr = hyp_params['lr_ada'][i]
							optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=0.1, eps=1e-8)
						elif model_type == 'eegnet':
							model = EEGNetv4(in_chans=n_chans, n_classes=n_classes, final_conv_length='auto', 
										 input_time_length=input_time_length, pool_mode='mean', F1=16, D=2, F2=32,
										 kernel_length=64, third_kernel_size=(8,4), drop_prob=drop_prob).create_network()
							lr = hyp_params['lr_adam'][i]
							optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, eps=1e-8, amsgrad=False)
						
						set_random_seeds(seed=20190629, cuda=cuda)
						
						if cuda:
							model.cuda()
							torch.backends.cudnn.deterministic = True
						model = torch.nn.DataParallel(model)
						log.info("%s model: ".format(str(model)))

						loss_function = loss_function
						model_loss_function = None

						#####Setup to run the selected model#####
						model_test = Experiment(model, train_set, valid_set, test_set=None, iterator=iterator,
												loss_function=loss_function, optimizer=optimizer,
												model_constraint=model_constraint, monitors=monitors,
												stop_criterion=stop_criterion, remember_best_column='valid_misclass',
												run_after_early_stop=True, model_loss_function=model_loss_function, cuda=cuda,
												data_type=data_type, subject_id=subject_id, model_type=model_type, 
												cropped=cropped, model_number=str(out_fold_num)) 

						model_test.run()
						model_loss = model_test.epochs_df['valid_loss'].astype('float')
						current_val_loss = current_loss(model_loss)
						loss_with_params[f"Fold_{in_fold_num}"][f"{drop_prob}/{loss_function}/{lr}"] = current_val_loss

		####Select and train optimized model#####
		df = pd.DataFrame(loss_with_params)
		df['mean'] = df.mean(axis=1) # compute mean loss across k-folds
		writer_df = f"results_folder\\results\\S{subject_id}\\{model_type}_parameters.xlsx"
		df.to_excel(writer_df)
		
		best_dp, best_loss, best_lr = df.loc[df['mean'].idxmin()].__dict__['_name'].split("/") # extract best param values
		if str(best_loss[10:13]) == 'nll':
			best_loss = F.nll_loss
		elif str(best_loss[10:13]) == 'cro':
			best_loss = F.cross_entropy
		
		print(f"Best parameters: dropout: {best_dp}, loss: {str(best_loss)[10:13]}, lr: {best_lr}")

		#####Train model on entire inner fold set#####
		torch.backends.cudnn.deterministic = True
		model = None
		#####Create outer-fold validation and test sets#####
		X_valid, X_test, y_valid, y_test = train_test_split(outer_fold, outer_labels, test_size=0.5, random_state=42, stratify=outer_labels)
		train_set = SignalAndTarget(inner_fold, inner_labels)
		valid_set = SignalAndTarget(X_valid, y_valid)
		test_set  = SignalAndTarget(X_test, y_test)


		if model_type == 'shallow':
			model =  ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes, input_time_length=input_time_length,
						 n_filters_time=60, filter_time_length=5, n_filters_spat=40, 
						 pool_time_length=50, pool_time_stride=15, final_conv_length='auto',
						 conv_nonlin=relu6, pool_mode='mean', pool_nonlin=safe_log, 
						 split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
						 drop_prob=0.1).create_network() #50 works better than 75
			
			optimizer = optim.Adadelta(model.parameters(), lr=2.0, rho=0.9, weight_decay=0.1, eps=1e-8) 
			
		elif model_type == 'deep':
			model = Deep4Net(in_chans=n_chans, n_classes=n_classes, input_time_length=input_time_length,
						 final_conv_length='auto', n_filters_time=20, n_filters_spat=20, filter_time_length=5,
						 pool_time_length=3, pool_time_stride=3, n_filters_2=20, filter_length_2=5,
						 n_filters_3=40, filter_length_3=5, n_filters_4=1500, filter_length_4=10,
						 first_nonlin=leaky_relu, first_pool_mode='mean', first_pool_nonlin=safe_log, later_nonlin=leaky_relu,
						 later_pool_mode='mean', later_pool_nonlin=safe_log, drop_prob=0.1, 
						 double_time_convs=False, split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
						 stride_before_pool=False).create_network()
			
			optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=0)
		elif model_type == 'eegnet':
			model = EEGNetv4(in_chans=n_chans, n_classes=n_classes, final_conv_length='auto', 
						 input_time_length=input_time_length, pool_mode='mean', F1=16, D=2, F2=32,
						 kernel_length=64, third_kernel_size=(8,4), drop_prob=0.1).create_network()
			optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0, eps=1e-8, amsgrad=False) 
			

		if cuda:
			model.cuda()
			torch.backends.cudnn.deterministic = True
			#model = torch.nn.DataParallel(model)
		
		log.info("Optimized model")
		model_loss_function=None
		
		#####Setup to run the optimized model#####
		optimized_model = op_exp(model, train_set, valid_set, test_set=test_set, iterator=iterator,
								loss_function=best_loss, optimizer=optimizer,
								model_constraint=model_constraint, monitors=monitors,
								stop_criterion=stop_criterion, remember_best_column='valid_misclass',
								run_after_early_stop=True, model_loss_function=model_loss_function, cuda=cuda,
								data_type=data_type, subject_id=subject_id, model_type=model_type, 
								cropped=cropped, model_number=str(out_fold_num))
		optimized_model.run()

		log.info("Last 5 epochs")
		log.info("\n" + str(optimized_model.epochs_df.iloc[-5:]))
		
		writer = f"results_folder\\results\\S{subject_id}\\{data_type}_{model_type}_{str(out_fold_num)}.xlsx"
		optimized_model.epochs_df.iloc[-30:].to_excel(writer)

		accuracy = 1 - np.min(np.array(optimized_model.class_acc))
		cv_scores.append(accuracy) # k accuracy scores for this param set. 
		
	#####Print and store fold accuracies and mean accuracy#####
	
	print(f"Class Accuracy: {np.mean(np.array(cv_scores))}")
	results_df = pd.DataFrame(dict(cv_scores=cv_scores,
								   cv_mean=np.mean(np.array(cv_scores))))

	writer2 = f"results_folder\\results\\S{subject_id}\\{data_type}_{model_type}_cvscores.xlsx"
	results_df.to_excel(writer2)
	return optimized_model, np.mean(np.array(cv_scores))


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
						level=logging.DEBUG, stream=sys.stdout)
torch.backends.cudnn.deterministic = True
subject_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15'] 
			   
model_types = ['shallow','deep', 'eegnet']
data_types  = ['words','vowels','all_classes']
cuda = False
cropped = '_nc'
parameters = dict(max_epochs=40, max_increase_epochs=30, batch_size=64) # training parameters

hyp_params = dict(drop_prob=[0.2,0.5,0.8],
				  lr_ada=[0.5,1.0,2.0],
				  lr_adam=[0.0001,0.001,0.01],
				  loss=[F.nll_loss, F.cross_entropy]) # model hyper-parameters


Totals = dict()
total_words, total_vowels = [], []
for subject_id in subject_ids:
	Totals[f"{subject_id}"] = dict()
	for model_type in model_types:
		Totals[f"{subject_id}"][f"{model_type}"] = dict()
		for data_type in data_types:
			
			run_model, scores = network_model(subject_id, model_type, data_type, cropped, cuda, parameters, hyp_params)

			Totals[f"{subject_id}"][f"{model_type}"][f"{data_type}"] = scores # mean accuracy for each subject
			
total_df = pd.DataFrame(Totals)
writer3 = f"C:/Users/cfcoo/OneDrive - Ulster University/Study_2/paper\\results\\totals.xlsx"
total_df.to_excel(writer3)
