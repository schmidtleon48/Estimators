'''
Topic:          Filter and Prediction Hidden Markov Model
Description:    Prediction and Wonham-Filter, 
                Transitionmatrix and meassurementmatrix independent from time and input
Autor:          Leon Schmidt
E-Mail:         leonjohannesschmidt@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

#Function predicts state
#input:     A- transitionmatrix
#           inital - Initial state
#           k - after how many timesteps predicted meassurement
def prediction(A, inital, k):
    A = np.transpose(A)
    Ak = np.identity(A.shape[0])
    
    i = 0
    while i < k:
        Ak = np.matmul(Ak, A)
        i += 1

    pred = np.matmul(Ak, inital)
    
    return pred

#Function estimates state based on meassurement
#input:     pred - predicted state
#           B - meassurement matrix
#           m - realized prediction
def wonham(pred, B, m):
    
    est = np.matmul(np.diag(B[:,m]), pred)
    est = est / np.sum(est)

    return est