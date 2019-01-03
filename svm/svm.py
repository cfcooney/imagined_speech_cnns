"""
Name: Ciaran Cooney
Date: 03/01/2019
Description: Script to extract mfcc features from imagined speech EEG data,
train a svm using a stratified, nested cross-validation scheme. Precision, recall, 
f-score and confusion matrices are calculated for each subject's data.
"""

import numpy as np
from utils import plot_confusion_matrix, data_wrangler
from wavelet_functions import mfcc_f 

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix 
import pandas as pd


def svm_model(subject_id, data_type):

	"""
	Initialize arrays for combined ground truth and predicitons
	for each fold. Used for combined stats.
	"""
	y_true_all = np.array([])
	y_pred_all = np.array([])

	# load data
	data, labels = data_wrangler(data_type, subject_id)

	"""
	Extract pre-computed windows for best performance. Can change.
	"""
	if data_type == 'words' or data_type == 'all_classes':
		data = data[:,:,768:1280]
	elif data_type == 'vowels':
		data = data[:,:,512:1024]


	"""
	Compute mfcc features using python_speech_features toolbox @:
	https://github.com/jameslyons/python_speech_features
	"""
	winlen=0.25
	winstep=0.05 
	sample_rate = 1024
	nfilt = 26
	nfft = 256 
	lowfreq = 2
	highfreq = 40
	
	subject_features = []
	for trial in data:
	    subject_features.append(mfcc_f(trial, sample_rate, winlen, winstep, nfilt, nfft, lowfreq, highfreq))
	data, labels = balanced_subsample(np.array(subject_features), labels) # downsampling to ensure equal number of targets.
	
	params = dict(C=[1, 10, 100, 1e3],
    		      gamma=[0.001, 0.1, 0.5, 1]) # Parameter search-space

	num_folds = 4
	data, _, labels, _ = train_test_split(data, labels, test_size=0, shuffle=True, random_state=42) #additional shuffle of data

	skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10) 
	cv_idx = 0
	cv_scores = []

	for inner_ind, outer_index in skf.split(data, labels):
	    inner_fold, outer_fold     = data[inner_ind], data[outer_index]
	    inner_labels, outer_labels = labels[inner_ind], labels[outer_index]
	    
	    cv_idx += 1
	    
	    Accuracy = dict()
	    
	    fold_num = 0
	    for train_idx, valid_idx in skf.split(inner_fold, inner_labels):
	        X_Train, X_val = inner_fold[train_idx], inner_fold[valid_idx]
	        y_train, y_val = inner_labels[train_idx], inner_labels[valid_idx]
	        fold_num += 1
	        Accuracy[f"Fold_{fold_num}"] = dict()
	        for c in params['C']:
	            for g in params['gamma']:
	                clf = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=g)
	                
	                clf.fit(X_Train, y_train)
	                pred = clf.predict(X_val)
	                accuracy = accuracy_score(pred, y_val)

	                Accuracy[f"Fold_{fold_num}"][f"{c}/{g}"] = accuracy # store accuracy with parameter values.
	        df = pd.DataFrame(Accuracy)
	        df['mean'] = df.mean(axis=1) # compute mean so best parameters can be selected.
	        writer_df = f"results_folder\\S{subject_id}\\parameters_mfcc_{cv_idx}.xlsx"
	        df.to_excel(writer_df)

	    best_c, best_g = df.loc[df['mean'].idxmax()].__dict__['_name'].split("/")
	    print(f"Parameters: {best_c}, {best_g}") # print best parameters

	    #####Train SVM on inner fold data with best parameter values#####
	    clf = SVC(kernel='rbf', class_weight='balanced', C=float(best_c), gamma=float(best_g))
	    
	    clf.fit(inner_fold, inner_labels)
	    pred = clf.predict(outer_fold)
	    accuracy = accuracy_score(pred, outer_labels)
	    cv_scores.append(accuracy) #all cross-validated accuracies
	    y_true_all = np.concatenate((y_true_all, outer_labels))
	    y_pred_all = np.concatenate((y_pred_all, pred))
	print(f"Mean Accuracy: {100*np.mean(np.array(cv_scores))}%")
	results_df = pd.DataFrame(dict(cv_scores=cv_scores,
								   cv_mean=np.mean(100*np.array(cv_scores))))
	writer2 = f"results_folder\\S{subject_id}_{data_type}_mfcc_cvscores.xlsx"
	results_df.to_excel(writer2)
	return cv_scores , y_true_all, y_pred_all


if __name__ == '__main__':
	all_cv = []

	subject_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15'] #15 subjects
	data_types = ['words', 'vowels', 'all_classes']
	labels_w = ['Arriba', 'Abajo', 'Adelante', 'Atrás', 'Derecha', 'Izquierda']
	labels_v = ['/a/', '/e/', '/i/', '/o/', '/u/']
	labels_c = ['/a/', '/e/', '/i/', '/o/', '/u/', 'Arriba', 'Abajo', 'Adelante', 'Atrás', 'Derecha', 'Izquierda']
	
	for data_type in data_types:
		stats = {'precision': [],
		             'recall': [],
		             'f-score': []}
		y_true_total = np.array([])
		y_pred_total = np.array([])
		for subject_id in subject_ids:

			cv_scores, y_true_all, y_pred_all = svm_model(subject_id, data_type)
			print(len(cv_scores))
			all_cv.append(np.mean(100*np.array(cv_scores)))
			all_results = pd.DataFrame(dict(cv_scores=all_cv))
			precision, recall, f_score, _ = precision_recall_fscore_support(y_true_all,y_pred_all)
			stats['precision'].append(np.mean(precision))
			stats['recall'].append(np.mean(recall))
			stats['f-score'].append(np.mean(f_score))
			y_true_total = np.concatenate((y_true_total, y_true_all))
			y_pred_total= np.concatenate((y_pred_total, y_pred_all))

		df = pd.DataFrame(stats, index=subject_ids)
		writer_df = f"C:\\Users\\sb00745777\\OneDrive - Ulster University\\Study_2\\results\\svm_{data_type}_stats1.xlsx"
		df.to_excel(writer_df)

		cm = confusion_matrix(y_true_total, y_pred_total)
		cm_filename = f"C:/Users/sb00745777/OneDrive - Ulster University/Study_2/results/misc/svm_{data_type}"

		if data_type == 'words':
			labels = labels_w
		elif data_type == 'vowels':
			labels = labels_v
		elif data_type == 'all_classes':
			labels = labels_c
		plot_confusion_matrix(cm, labels, filename=cm_filename, normalize=True)
		writer3 = f"results_folder\\mfcc_cvscores.xlsx"
		all_results.to_excel(writer3)
