"""
Name: Ciaran Cooney
Date: 23/08/2018
Description: Script for importing imagined speech EEG data (vowels and words) along with corresponding class labels. 
This includes removal of all overt speech EEG data. Data is stored as pickles in subject folders. The imagined speech
EEG dataset is described in [1] and downloaded to local folder before running this script.

[1] Coretto, G.A.P., Gareis, I.E. and Rufiner, H.L., 2017, January. Open Access database of EEG signals recorded 
	during imagined speech. In 12th International Symposium on Medical Information Processing and Analysis 
	(Vol. 10160, p. 1016002). International Society for Optics and Photonics.
"""

import scipy.io as spio
import numpy as np
import pickle
import os
from utils import load_pickle


def extract_imagined_speech(data_type, ranges):

		dir = ('..//imagined_speech/')
		speech_index, class_index, epoch = 24576, 24577, 4096 # provided in dataset readme file
		for folder in os.listdir(dir):
			if not folder.endswith(".txt") and not folder.endswith(".xlsx") and not folder.endswith('ica'):
				for file in os.listdir(dir + folder):
					if file.endswith("EEG.mat"):
						print(f"Loading EEG data for subject {folder}...")
						data = spio.loadmat(dir + folder + '/' + file)['EEG']
						imagined, data, labels, eeg = ([] for i in range(4))

						#####Remove overt speech from dataset#####
						[imagined.append(i) for i in data if i[speech_index] == 1.0]
						imagined = np.array(imagined)
						print(f"{folder} 'total classes: {imagined.shape[0]}")

						#####Extract data-type, i.e., vowels or words#####
						[data.append(j) for j in imagined if j[class_index] in range(ranges[0],ranges[1])]
						for i in range(len(data)):
							eeg.append(data[i][0:speech_index])
						data = np.array(data)
						eeg = np.array(eeg)

						[labels.append(data[k][class_index]) for k in range(len(data))]
						labels = np.array(labels)
						print(f"{folder} {data_type} classes: {labels.shape[0]}")
						
						index, eeg_format, eeg_2d = ([] for i in range(3))
						[index.append(i) for i in range(0,speech_index,epoch)]

						for j in eeg:
							eeg_index = []
							[eeg_index.append(j[index[k]:index[k]+epoch]) for k in range(len(index))]
							eeg_format.append(eeg_index)
						eeg_format = np.array(eeg_format)
						print(f"{folder} 3d data shape: {eeg_format.shape}") # labels*channels*samples

						[eeg_2d.append(i) for i in eeg_format]
						eeg_2d = np.array(eeg_2d)
						eeg_2d = np.reshape(eeg_2d,(6,(eeg_format.shape[0]*eeg_format.shape[2])))
						print(folder + ' 2d data shape: ' + str(eeg_2d.shape)) # channels*samples

						print(f"Saving EEG and labels for subject {folder}...")
						pickle_file = dir + folder + '/'+'imagined_{data_type}.pickle'

						try:
							f = open(pickle_file, 'wb')
							save = {
									'imagined_{data_type}': data,
									'imagined_labels': labels,
									'imagined_eeg_3d': eeg_format,
									'imagined_eeg_2d': eeg_2d
									}
							pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
							f.close()
						except Exception as e:
							print('Unable to save data to', pickle_file, ':', e)
							raise
					
					
if __name__ == "__main__":

	data_ranges = dict(words=(6,12),
					   vowels=(1,6))

	for data_type, ranges in data_ranges.items():
		
		extract_imagined_speech(data_type, ranges)
