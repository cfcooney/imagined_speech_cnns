import pickle
import os
import numpy as np 
import pandas as pd

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
        with open(data_folder + vowels_file, 'rb') as f:
            file = pickle.load(f)
        v_data = file['raw_array'][:][0]
        v_labels = file['labels']
    return w_data, v_data, w_labels, v_labels


