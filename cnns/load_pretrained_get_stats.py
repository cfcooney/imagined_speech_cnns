"""
Name: Ciaran Cooney
Date: 13/01/2019
Description: Script to load a pre-trained pytorch cnn model, test the model on 
the test dataset in a 4-fold cross-validation scheme, and collect statistics 
on the models performance. These include precision, recall and f-score, to be stored
locally as csv files
"""

import torch
from utils import load_subject_eeg, eeg_to_3d, balanced_subsample, data_wrangler, predict, print_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import BalancedBatchSizeIterator
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import itertools
from eegnet import EEGNetv4




def get_stats(model_type, data_type):
	subject_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15'] # 15 subjects
	accuracies = []
	batch_size = 64
	iterator = BalancedBatchSizeIterator(batch_size=batch_size)

	#####Instantiate variables for results#####
	y_true_all = np.array([])
	y_pred_all = np.array([])
	stats = {'precision': [],
             'recall': [],
             'f-score': []}

	for subject_id in subject_ids:

		#####Collect and format the data correctly#####
		data, labels = data_wrangler(data_type, subject_id)
		data, labels = balanced_subsample(data, labels)
		if data_type == 'words' or 'all_classes':
		    data = data[:,:,768:1280]
		elif data_type == 'vowels':
		    data = data[:,:,512:1024]
		num_folds = 4

		data, _, labels, _ = train_test_split(data, labels, test_size=0, random_state=42) #shuffle the data/labels
		skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)

		test_data = [] # list for appending all test_data folds
		y_true = [] # lsit for appending true class labels
		y_true_a = np.array([])
		sub_acc = [] # list for appending all of the subjects accuracies

		#####split data into inner/outer folds and extract test sets#####
		for inner_ind, outer_index in skf.split(data, labels):
			
			inner_fold, outer_fold     = data[inner_ind], data[outer_index]
			inner_labels, outer_labels = labels[inner_ind], labels[outer_index]
			outer_fold = outer_fold.reshape((outer_fold.shape[0],6,512,1)) # expected format
			_, X_test, _, y_test = train_test_split(outer_fold, outer_labels, test_size=0.5, random_state=42, stratify=outer_labels)
			
			test_data.append(X_test) # test data
			y_true.append(y_test) # test labels

		
		models = [0,1,2,3]
		y_pred = np.array([])

		#####Load pytorch model and make predictions#####
		for model_num in models:
		   
		    if data_type == 'all_classes':
		    	data_type1 = 'combined'
		    else:
		    	data_type1 = data_type

		    model_file = f"results_folder/stored_models/{model_type}_{data_type1}/S{subject_id}/"
		    model = None # avoids potential duplication
		    model = torch.load(model_file + f"{data_type}_{model_type}model_nc_{model_num+1}.pt", map_location={'cuda:0': 'cpu'}) #load model
		    prediction = predict(model, test_data[model_num], 64, iterator)
		    accuracy = accuracy_score(prediction, y_true[model_num])
		    print(f"Accuracy: {accuracy}")
		    sub_acc.append(accuracy)
		    accuracies.append(accuracy)
		    y_pred = np.concatenate((y_pred, prediction))
		print(f"Subject accuracy: {np.mean(np.array(sub_acc)) * 100} %")

		for y in y_true:
			y_true_a = np.concatenate((y_true_a, y))
		
		#####Gather statistics and add to dict#####
		precision, recall, f_score,_ = precision_recall_fscore_support(y_true_a,y_pred)
		stats['precision'].append(np.mean(precision))
		stats['recall'].append(np.mean(recall))
		stats['f-score'].append(np.mean(f_score))
		y_true_all = np.concatenate((y_true_all, y_true_a))
		y_pred_all = np.concatenate((y_pred_all, y_pred))
	print(f"Average Accuracy: {np.mean(np.array(accuracies)) * 100} %")

	df = pd.DataFrame(stats, index=subject_ids)
	writer_df = f"results_folder\\results\\{model_type}_{data_type}_stats.xlsx"
	df.to_excel(writer_df)

	return y_true_all, y_pred_all




if __name__ == '__main__':
	
	#####Instantiate models, data and labels#####
	labels_w = ['Arriba', 'Abajo', 'Adelante', 'Atrás', 'Derecha', 'Izquierda']
	labels_v = ['/a/', '/e/', '/i/', '/o/', '/u/']
	labels_c = ['/a/', '/e/', '/i/', '/o/', '/u/', 'Arriba', 'Abajo', 'Adelante', 'Atrás', 'Derecha', 'Izquierda']
	model_types = ['shallow','deep','eegnet']
	data_types = ['vowels', 'words','all_classes']

	for model_type in model_types:
		for data_type in data_types:
			y_true_all, y_pred_all = get_stats(model_type, data_type) #run the main function
			cm = confusion_matrix(y_true_all, y_pred_all)
			cm_filename = f"results_folder/results/misc/{model_type}_{data_type}" #folder to save to
			
			if data_type == 'words':
				labels = labels_w
			elif data_type == 'vowels':
				labels = labels_v
			elif data_type == 'all_classes':
				labels = labels_c
			print_confusion_matrix(cm, labels, filename=cm_filename, normalize=True)
