import numpy as np
import matplotlib.pyplot as plt

#define system
iteration = 9
A = np.array([[1.1, 0],[0,1.2]])
H = np.array([1,2])
inital = np.array([[-3], [-3]])
noise_state = 0.2 * np.random.randn(2,9)
noise_mess = 0.6 * np.random.randn(9)

#propagation through system
state_mom = inital 
i = 0
state = inital
output = []
while i < iteration:


    state_mom = A @ state_mom + noise_state[:, i].reshape(2,1)
    output_mom = H @ state_mom

    state = np.concatenate((state, state_mom), axis = 1)
    output.append(output_mom[0])
    i += 1


steps = np.arange(9)
for k in steps:
    plt.clf() # Clear the current figure
    
    plt.scatter(state[0, k], state[1, k]) # Calculate and plot all you want
    plt.xlim(-20,20)
    plt.ylim(-50, 20)
#    plt.fill_between(x, y, y+1, facecolor='C0', alpha=0.4)

    plt.draw()
    plt.pause(0.4) # Has to pause for a non zero time


plt.show() # When all is done
