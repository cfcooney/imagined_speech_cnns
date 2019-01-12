"""
Name: Ciaran Cooney
Date: 11/11/2018
Description: Script for training CNNs on imagined speech EEG data which has been 
divided into two classes, words and vowels. Data is loaded, separated into the 
two classes and labels of 0 and 1 assigned. 

This means that a model is trained on 'arriba' vs '/a/', then 'arriba' vs /e/, ..., 'Izquierda' vs '/u/'
"""

import logging 
import os.path
import time 
from collections import OrderedDict 
import sys
from utils import load_subject_eeg, eeg_to_3d, return_indices 
import pandas as pd 
import numpy as np 
import torch.nn.functional as F 
from torch import optim 

from braindecode.models.deep4 import Deep4Net 
from braindecode.experiments.experiment import Experiment # performs one experiment on training, validation and test
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator # ensures balanced batch sizes 
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet 
from braindecode.datautil.splitters import split_into_two_sets # split by fraction or number 
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint # applies L2 norm to weights
from braindecode.torch_ext.util import set_random_seeds, np_to_var # transform np array to torch-tensor 
from braindecode.mne_ext.signalproc import mne_apply 
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize # perform exponential running standardisation 
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.model_selection import train_test_split
from braindecode.torch_ext.optimizers import AdamW

log = logging.getLogger(__name__) 

def network_model(model, train_set, test_set, valid_set, n_chans, input_time_length, cuda):
	
	max_epochs = 30 
	max_increase_epochs = 10 
	batch_size = 64 
	init_block_size = 1000

	set_random_seeds(seed=20190629, cuda=cuda)

	n_classes = 2 
	n_chans = n_chans
	input_time_length = input_time_length

	if model == 'deep':
		model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
						 final_conv_length='auto').create_network()

	elif model == 'shallow':
		model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
								final_conv_length='auto').create_network()

	if cuda:
		model.cuda()

	log.info("%s model: ".format(str(model))) 

	optimizer = AdamW(model.parameters(), lr=0.00625, weight_decay=0)

	iterator = BalancedBatchSizeIterator(batch_size=batch_size) 

	stop_criterion = Or([MaxEpochs(max_epochs),
						 NoDecrease('valid_misclass', max_increase_epochs)])

	monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

	model_constraint = None
	print(train_set.X.shape[0]) 

	model_test = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
							loss_function=F.nll_loss, optimizer=optimizer,
							model_constraint=model_constraint, monitors=monitors,
							stop_criterion=stop_criterion, remember_best_column='valid_misclass',
							run_after_early_stop=True, cuda=cuda)

	model_test.run()
	return model_test 

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
						level=logging.DEBUG, stream=sys.stdout)

Subject_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']

for sbj in Subject_ids: 
	subject_id = sbj
	model = 'shallow' 
	cuda = False
	#writer = 'results_folder\\results\\S' + subject_id + '\\multicom_results.xlsx'
	w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
	n_chan, epoch = len(w_data), 4096 
	w_data = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan)
	v_data = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan)

	#####Compute indices for each word and vowel in the dataset#####
	word_id = dict(Arriba=6, Abajo=7, Adelante=8, Atrás=9, Derecha=10, Izquierda=11)
	vowel_id = dict(a=1, e=2, i=3, o=4, u=5)
	word_indices = return_indices(word_id, w_labels)
	vowel_indices = return_indices(vowel_id, v_labels)
	
	n_chans = int(w_data.shape[1])
	input_time_length = w_data.shape[2]
	
	#####Train and test on each word/vowel combination#####
	for i, p in enumerate(word_id):
		
		print("Working on Subject: " + subject_id + ", Word: " + p)
		writer = f"results_folder\\results\\S{subject_id}\\{p}.xlsx"
		features_w = w_data[word_indices[i]]

		scores_list = []
		scores_all = []
		vowels_results = pd.DataFrame()
		for j in range(len(vowel_indices)):
			
			#####Combine training data#####
			features_v = v_data[vowel_indices[j]]
			features = np.concatenate((features_w, features_v)).astype(np.float32)

			w_labels = np.zeros(features_w.shape[0])
			v_labels = np.ones(features_v.shape[0])
			labels = np.concatenate((w_labels, v_labels)).astype(np.int64)

			valid_set_fraction = 0.3 

			X_Train, X_Test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
			y_train = (y_train).astype(np.int64)

			train_set = SignalAndTarget(X_Train, y_train)
			test_set = SignalAndTarget(X_Test, y_test)

			train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=1-valid_set_fraction)
			
			run_model = network_model(model, train_set, test_set, valid_set, n_chans, input_time_length, cuda) 
			log.info('Last 10 epochs')
			log.info("\n" + str(run_model.epochs_df.iloc[-10:]))
			
			vowels_results.append(run_model.epochs_df.iloc[-10:])
			#vowels_results = vowels_results.append(pd.DataFrame())
			#print(vowels_results)
			#run_model.epochs_df.iloc[-10:].to_excel(writer,'sheet%s' %str(j+1))

		print(f"Saving classification results for Subject: {sbj}")
		vowels_results.to_excel(writer) # saving results for one word vs all 5 vowels
