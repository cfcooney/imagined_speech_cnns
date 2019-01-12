"""
Name: Ciaran Cooney
Date: 27/09/2018
Description: Script for computing relative wavelet energy from imagined speech
EEG signals. Can be adapted for other EEG inputs.
"""

import numpy as np 
import pickle
from scipy.signal import decimate as dec
import pywt._multilevel 
from utils import load_pickle, eeg_to_3d 
import os 



def compute_wavelets(dir, filename, level):

	for folder in os.listdir(dir):
		if not folder.endswith(".txt") and not folder.endswith(".xlsx") and not folder.endswith("ica"):
			new_folder = folder + '/post_ica'
			file, _ = load_pickle(dir, new_folder, filename)
			data = file['raw_array'][:][0]
			labels = file['labels']

			n_chan, n_events, epoch = len(data), len(labels), 4096 

			data = eeg_to_3d(data, epoch, n_events, n_chan) # convert 2d array to events*channels*samples matrix
			data = dec(data, 8)

			subject_features = []
			for event in data:
				ch_features =[]
				for ch in event:

					wt = pywt._multilevel.wavedec(ch, db, level=level)
					#####approximation and detail coefficients names#####
					cf_names = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']

					Ej = dict()
					Ejt = dict()
					for i in range(len(wt)):
						cf_list = []
						[cf_list.append(np.absolute(c)**2) for c in wt[i]]
						Ej[cf_names[i]] = cf_list 

					for c, k in enumerate(Ej):
						Ejt[cf_names[c]] = np.sum(Ej[k])

					Etot = 0
					for _, k in enumerate(Ejt):
						Etot = Etot + Ejt[k]

					rwe = []

					for _, k in enumerate(Ejt):
						rwe.append(Ejt[k] / Etot)
					rwe = rwe[:-1]

					ch_features.append(rwe)
				features = np.array(ch_features).reshape((30))
				subject_features.append(features)
			save_file = dir + new_folder + '/' + 'wavelet_features_{data_type}.pickle'
			f = open(save_file, 'wb')
			save = {'wavelet_features': subject_features,
						   'labels': labels}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()

if __name__ == "__main__":

	data_types = ['words, vowels']
	db = 'db4' # daubechies value
	dir = '..//imagined_speech/'
	for data_type in data_types:
		filename = 'raw_array_{data_type}_ica.pickle'

		compute_wavelets(dir, filename, 5, db)






  