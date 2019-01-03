"""
Name: Ciaran Cooney
Date: 19/09/2018
Description: Functions for use in computing features for decoding imagined speech
from EEG. Relative wavelet energy and mel frequency cepstral coefficients.
"""

import pywt._multilevel
import numpy as np
from utils import reverse_coeffs, load_pickle 
from scipy.fftpack import dct
import python_speech_features as psf
from python_speech_features.base import get_filterbanks
from python_speech_features import base

def sumup(data, interval):
    data = data.reshape(1,len(data))
    M,N = data.shape
    N1 = np.floor(N / interval)
    y = np.zeros((M,int(N1)))

    for k in range(1,M+1):
        for i in range(1,int(N1+1)):
            mlist = list()
            for n in range(1,int(interval+1)):
                mlist.append(data[k-1, int((i-1)*interval+(n-1))])
            m = np.sum(mlist)
            y[k-1,i-1] = m
    return y

def relative_energy(data):
    m,n = data.shape
    
    total = np.sum(data)
    for i in range(n):
        for j in range(m):
            data[j,i] = data[j,i] / total
    return data

def wavelets_f(event):
    ch_features = []
    for ch in event:
        """Discrete Wavelet Transform"""
        wt = pywt._multilevel.wavedec(ch, 'db4', level=5) # daubechies wavelets, level 5
        #a5,d5,d4,d3,d2,d1 = wt possible problem here
        l = []
        [l.append(i.shape[0]) for i in wt]
        l.append(512)

        wt = np.array(wt)
        coeffs = []
        [[coeffs.append(j) for j in wt[i]] for i in range(len(wt))]

        """Relative Wavelet Energy"""
        N1 = 256 # interval must be the number of k*(2^levels),where k is an integer.
        N2 = len(l)-2
        a=np.zeros((N2,l[N2]))
        c = np.array(coeffs)
        c = c*c

        for k in range(N2):
            a[k][0:l[1+k]] = c[np.sum(l[0:k+1]):np.sum(l[0:k+2])] #possible problem here
        N3=N1/2**(N2)
        N4=np.floor(l[1]/N3)
        b=np.zeros((N2,int(N4)))

        for k in range(1,N2+1):
            k1 = np.log2(N3)+k-1
            x=sumup(a[k-1,:],2**k1)
            b[k-1, 0:int(np.ceil(l[k]/2**k1)-1)] = x[0][0:int(np.ceil(l[k]/2**k1)-1)]

        _,y = reverse_coeffs(b[:5,0], N2)

        yout=relative_energy(y)

        ch_features.append(yout)
    return np.array(ch_features).reshape((30))



def mfcc_f(trial, sample_rate, winlen, winstep, nfilt, nfft, lowfreq, highfreq):
    winfunc=lambda x:np.ones((x,))
    n_chans = trial.shape[1]
    ch_features = []
    for ch in trial:
        frames = psf.sigproc.framesig(ch, winlen*sample_rate, winstep*sample_rate, winfunc)
        fb, melpoints = get_filterbanks(nfilt,nfft,sample_rate,lowfreq,highfreq)
        
        pspec = psf.sigproc.powspec(frames,nfft)
        energy = np.sum(pspec,1) # this stores the total energy in each frame
        energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log
        feat = np.dot(pspec,fb.T) # compute the filterbank energies
        feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log
        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:12] #13 - numceps
        feat = base.lifter(feat,22)

        ch_features.append(np.reshape(feat,((feat.shape[0]*feat.shape[1]))))
    return np.array(ch_features).reshape((np.array(ch_features).shape[0]*np.array(ch_features).shape[1]))
