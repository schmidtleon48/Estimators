'''
Topic:          Filter and Prediction Hidden Markov Model
Description:    Demonstration 
Autor:          Leon Schmidt
E-Mail:         leonjohannesschmidt@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import hmm

#Generate transitionsmatrix and meassuremant matrix
n_x = 10
n_out = 5
k = 10

A = np.random.rand(n_x, n_x)
norm_a = 1/ np.sum(A, axis=1)
A = np.multiply(A, norm_a[:, np.newaxis])

i = 0
B = np.zeros((n_x, n_out))
while i < n_out:
    B_ind = np.random.randint(0, n_x , int( 0.7 * n_x ))
    B[B_ind, i] = 1
    i += 1

norm_b = 1/ np.sum(B, axis=0)
B = np.multiply(B.transpose(), norm_b[:, np.newaxis])
B = np.transpose(B)

#first state most likely
inital = np.zeros(n_x)
inital[0] = 1

#predicte states in hidden markov model
x = np.arange(k)
y = np.arange(n_x)
z = np.zeros((n_x, k))
for i in x:
    z[:,i] = hmm.prediction(A, inital, i)

#visualize prediction
plt.pcolormesh(x, y, z)
plt.colorbar()
plt.title('Flow of states')
plt.xlabel('Time')
plt.ylabel('States')
plt.show()

#estimate state 
z_new = np.zeros((n_x, 2))
z_new[:,0] = z[:, k-1]
z_new[:,1] = hmm.wonham(z[:, k-1], B, 0)

#visualize estimation
plt.pcolormesh(['predicted', 'estimated'], y, z_new)
plt.colorbar()
plt.title('After estimation of states')
plt.ylabel('States')
plt.show()