'''
Topic:          Kalman Filter
Description:    Implementation of the Kalman Filter
Autor:          Leon Schmidt
E-Mail:         leonjohannesschmidt@gmail.com
'''

import numpy as np

#Function calculates kalman filter
#input:     Cp - Covariance Matrix of predicted state
#           Cy - Covariance Matrix of messuremant noise
#           H -  Meassurement Matrix
def kalman(Cp, Cy, H):
    K = Cp @ H.T @ np.linalg.inv(H @ Cp @ H.T + Cy)
    return K