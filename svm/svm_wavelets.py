import numpy as np
from utils import load_subject_eeg, eeg_to_3d, balanced_subsample, reverse_coeffs 
from wavelet_functions import sumup, relative_energy, wavelets_f 

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

def data_wrangler(data_type, subject_id):
	epoch = 4096
	if data_type == 'words':
		data, labels = load_subject_eeg(subject_id, vowels=False)
		n_chan = len(data)
		data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
		labels = labels.astype(np.int64)
	elif data_type == 'vowels':
		_, data, _, labels = load_subject_eeg(subject_id, vowels=True)
		n_chan = len(data)
		data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
		labels = labels.astype(np.int64)
	elif data_type == 'all_classes':
		w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
		n_chan = len(w_data)
		words = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan).astype(np.float32)
		vowels = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan).astype(np.float32)
		data = np.concatenate((words, vowels), axis=0)
		labels = np.concatenate((w_labels, v_labels), axis=0).astype(np.int64)
	
	x = lambda a: a * 1e6
	data = x(data)
	
	if data_type == 'words': # zero-index the labels
		labels[:] = [x - 6 for x in labels]
	elif (data_type == 'vowels' or data_type == 'all_classes'):
		labels[:] = [x - 1 for x in labels]

	return data, labels


def svm_model(subject_id, data_type):
	data, labels = data_wrangler(data_type, subject_id)

	if data_type == 'words' or data_type == 'all_classes':
		data = data[:,:,768:1280]
	elif data_type == 'vowels':
		data = data[:,:,512:1024]
	
	#####Compute Wavelet features#####
	subject_features = []
	for trial in data:
	    subject_features.append(wavelets_f(trial))
	data, labels = balanced_subsample(np.array(subject_features), labels)
	
	params = dict(C=[1, 10, 100, 1e3],
    		      gamma=[0.001, 0.1, 0.5, 1])

	num_folds = 4
	data, _, labels, _ = train_test_split(data, labels, test_size=0, shuffle=False, random_state=42)

	skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=10)
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
	                Accuracy[f"Fold_{fold_num}"][f"{c}/{g}"] = accuracy
	        df = pd.DataFrame(Accuracy)
	        df['mean'] = df.mean(axis=1)
	        writer_df = f"C:\\Users\\sb00745777\\OneDrive - Ulster University\\Study_2\\results\\nested_results\\S{subject_id}\\parameters_svm_{cv_idx}.xlsx"
	        df.to_excel(writer_df)

	    best_c, best_g = df.loc[df['mean'].idxmax()].__dict__['_name'].split("/")
	    print(f"Parameters: {best_c}, {best_g}")
	    
	    clf = SVC(kernel='rbf', class_weight='balanced', C=float(best_c), gamma=float(best_g))
	    clf.fit(inner_fold, inner_labels)
	    pred = clf.predict(outer_fold)
	    accuracy = accuracy_score(pred, outer_labels)
	    cv_scores.append(accuracy)
	print(f"Mean Accuracy: {100*np.mean(np.array(cv_scores))}%")
	results_df = pd.DataFrame(dict(cv_scores=cv_scores,
								   cv_mean=np.mean(100*np.array(cv_scores))))
	writer2 = f"C:\\Users\\sb00745777\\OneDrive - Ulster University\\Study_2\\results\\baseline\\S{subject_id}_{data_type}_svm_cvscores.xlsx"
	results_df.to_excel(writer2)


if __name__ == '__main__':
	

	subject_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
	data_types = ['words','vowels', 'all_classes']
	for subject_id in subject_ids:
		for data_type in data_types:
			svm_model(subject_id, data_type)
