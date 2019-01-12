"""
Name: Ciaran Cooney
Date: 12/01/2019
Description: Script for computing Bit-rate and information
transfer rate. Use case here is for imagined speech EEG.
N = number of classes
T = length of trial in seconds
P = probablilty of correct prediction, int or list of ints
"""

import numpy as np 

N = 5 
T = 0.5

"""For single accuracy score"""
# B = (np.log2(N) + (P * np.log2(P)) + (1-P)*np.log2((1-P) / (N-1)))
# ITR = B / (T/60)

"""Iterable formulation for lists of results"""
P = [0.6968, 0.5843, 0.7690] # example results

def itr(N, P, T):
    assert type(P) is list, "list of probabilities required!"
    bits, itrs = [], []
    for p in P:
        B = (np.log2(N) + (p * np.log2(p)) + (1-p)*np.log2((1-p) / (N-1)))
        ITR = B / (T/60)
        bits.append(B)
        itrs.append(ITR)
    return bits, itrs
    
bits, itrs = itr(N, P, T)
for itr in itrs:
    print(f"information transfer rate: {itr} bits/min")
    print(f"Bits = {bits}")
