"""
Name: Ciaran Cooney
Date: 03/01/2019
Description: Utility functions for preprocessing EEG imagined speech
data. Includes functions for loading torch model for new predictions 
and plotting confusion matrices with predicited labels.
"""

import pickle
import os
import numpy as np 
import pandas as pd

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.experiments.monitors import compute_pred_labels_from_trial_preds
import torch as th

import matplotlib.pyplot as plt

import numpy as np
import itertools

def load_pickle(direct, folder, filename):
	
		for file in os.listdir(direct + folder):
			if file.endswith(filename):
				pickle_file = (direct + folder + '/' + file)
				with open(pickle_file, 'rb') as f:
					file = pickle.load(f)

				return file, pickle_file

def create_events(data, labels):
	events = []
	x = np.zeros((data.shape[0], 3))
	for i in range(data.shape[0]):
		x[i][0] = i 
		x[i][2] = labels[i]
	[events.append(list(map(int, x[i]))) for i in range(data.shape[0])]
	return np.array(events)

def reverse_coeffs(coeffs, N):
	""" Reverse order of coefficients in an array."""
	idx = np.array([i for i in reversed(range(N))])
	coeffs = coeffs[idx]
	coeffs = coeffs.reshape((N,1))
	z = np.zeros((N,1))
	return np.append(coeffs, z, axis=1) , coeffs

def class_ratios(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_weight = dict()
    for i in range(len(unique)):
       class_weight[unique[i]] = len(labels) / (len(unique)*counts[i])
    return class_weight

def classification_report_csv(report, output_file):
    """
    Saves sklearn classification report as csv file.
    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(output_file + '.csv', index = False)

def load_features(direct, dict_key1, dict_key2=None):
    with open(direct, 'rb') as f:
        file = pickle.load(f)
    if dict_key2 == None:
        return np.array(file[dict_key1])
    else:
        return np.array(file[dict_key1]), np.array(file[dict_key2])

def short_vs_long(features, labels, split, event_id):
    """Function for multilabel data into binary-class sets i.e.,
       short words and long words
    """
    short, long, s_idx, l_idx, s_features, l_features = ([] for i in range(6))
    
    [short.append(event_id[i]) for i in event_id if len(i) <= split]
    [long.append(event_id[i]) for i in event_id if len(i) > split]
    
    [s_idx.append(i) for i, e in enumerate(labels) if e in short]
    [l_idx.append(i) for i, e in enumerate(labels) if e in long]
    
    [s_features.append(e) for i, e in enumerate(features) if i in s_idx]
    [l_features.append(e) for i, e in enumerate(features) if i in l_idx]
    
    s_labels = np.zeros(np.array(s_features).shape[0])
    l_labels = np.ones(np.array(l_features).shape[0])

    features = np.concatenate((s_features, l_features))
    labels = np.concatenate((s_labels,l_labels))
    
    return s_features, l_features, s_labels, l_labels, features, labels 

def eeg_to_3d(data, epoch_size, n_events,n_chan):
    """
    function takes 2D EEG array and returns 3D array.
    """
    idx, a, x = ([] for i in range(3))
    [idx.append(i) for i in range(0,data.shape[1],epoch_size)]
    
    for j in data:
        [a.append([j[idx[k]:idx[k]+epoch_size]]) for k in range(len(idx))]
        
    for i in range(n_events):
        x.append([a[i],a[i+n_events],a[i+n_events*2],a[i+n_events*3],a[i+n_events*4],a[i+n_events*5]])
    
    return np.reshape(np.array(x),(n_events,n_chan,epoch_size))

def return_indices(event_id, labels):
    indices = []
    for _, k in enumerate(event_id):
        idx = []
        for d, j in enumerate(labels):
            if event_id[k] == j:
                idx.append(d)
        indices.append(idx)
    return indices

def load_subject_eeg(subject_id, vowels):
    """ returns eeg data corresponding to words and vowels 
        given a subject identifier.
    """

    data_folder = 'C:\\Users\\sb00745777\\OneDrive - Ulster University\\Study_2\\imagined_speech/S{}/post_ica/'.format(subject_id)
    data_folder1 = 'C:\\Users\\cfcoo\\OneDrive - Ulster University\\Study_2\\imagined_speech/S{}/post_ica/'.format(subject_id)
    words_file = 'raw_array_ica.pickle'
    vowels_file = 'raw_array_vowels_ica.pickle'
    
    try:
        with open(data_folder + words_file, 'rb') as f:
            file = pickle.load(f)
    except:
        print("Not on PC! Attempting to load from laptop.")
        with open(data_folder1 + words_file, 'rb') as f:
            file = pickle.load(f)
            
    w_data = file['raw_array'][:][0]
    w_labels = file['labels']
    if vowels == False:
        return w_data, w_labels

    elif vowels:
        try:
            with open(data_folder + vowels_file, 'rb') as f:
                file = pickle.load(f)
        except:
            with open(data_folder1 + vowels_file, 'rb') as f:
                file = pickle.load(f)
        v_data = file['raw_array'][:][0]
        v_labels = file['labels']
    return w_data, v_data, w_labels, v_labels

def balanced_subsample(features, targets, random_state=12):
    """
    function for balancing datasets by randomly-sampling data
    according to length of smallest class set.
    """
    from sklearn.utils import resample
    unique, counts = np.unique(targets, return_counts=True)
    unique_classes = dict(zip(unique, counts))
    mnm = len(targets)
    for i in unique_classes:
        if unique_classes[i] < mnm:
            mnm = unique_classes[i]

    X_list, y_list = [],[]
    for unique in np.unique(targets):
        idx = np.where(targets == unique)
        X = features[idx]
        y = targets[idx]
        
        X1, y1 = resample(X,y,n_samples=mnm, random_state=random_state)
        X_list.append(X1)
        y_list.append(y1)
    
    balanced_X = X_list[0]
    balanced_y = y_list[0]
    
    for i in range(1, len(X_list)):
        balanced_X = np.concatenate((balanced_X, X_list[i]))
        balanced_y = np.concatenate((balanced_y, y_list[i]))

    return balanced_X, balanced_y



def predict(model, X_test, batch_size, iterator, threshold_for_binary_case=None):
    """
    Load torch model and make predictions on new data.
    """
    all_preds = []
    with th.no_grad():
        for b_X, _ in iterator.get_batches(SignalAndTarget(X_test, X_test), False):
            b_X_var = np_to_var(b_X)
            all_preds.append(var_to_np(model(b_X_var)))

        pred_labels = compute_pred_labels_from_trial_preds(
                    all_preds, threshold_for_binary_case)
    return pred_labels


def plot_confusion_matrix(cm, classes,filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    Code for confusion matrix extracted from here:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(1, figsize=(9, 6))
    #ax = plt.add_subplot(111)
    plt.tick_params(labelsize='large')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize='large', fontname='sans-serif')
    plt.xlabel('Predicted label', fontsize='large', fontname='sans-serif')
    fig.savefig(filename + '.jpg', bbox_inches='tight')
    return(fig)

def data_wrangler(data_type, subject_id):
    """
    Function to return EEG data in format #trials*#channels*#samples.
    Also returns labels in the range 0 to n-1.
    """
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
