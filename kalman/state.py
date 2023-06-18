'''
Topic:          Kalman Filter
Description:    Demonstration of the Kalman Filter
Autor:          Leon Schmidt
E-Mail:         leonjohannesschmidt@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import kalman

#define system
iteration = 30
var_mess = 2
var_sys = 0.2
var_state_init = 0.1

A = np.array([[1.1, 0],[0,1.1]])
B = np.identity(2)
H = np.array([[1,0], [0,1]])
inital = np.array([[-3], [-3]])
noise_state = var_sys * np.random.randn(2,iteration +2)
noise_mess = var_mess * np.random.randn(iteration + 2)
cov_nstate = np.array([[var_sys**2, 0],
                      [0,var_sys**2]])
cov_nmess = np.array([[var_mess**2,0],[0,2 * var_mess**2]])
cov_state = np.array([[var_state_init,0],
                     [0,var_state_init]])

#propagation and visualization
state_mom = inital 
state_mom_est = inital
i = 0
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
plt.pause(12)
while i <= iteration:
    #calculate ellipse
    a = cov_state[0,0]
    b = cov_state[0,1]
    c = cov_state[1,1]

    width = a + c + np.sqrt((a - c)**2  + 4 * b**2)
    height = a + c - np.sqrt((a - c)**2  + 4 * b**2)
    theta = 0 
    if b == 0 and a >= c:
        theta = 0
    elif b == 0 and a < c:
        theta = np.pi / 2
    else:
        theta = np.arctan2(width/ 2 -a, b)

    angle = np.rad2deg(theta) 


    #plot current state
    ellipse = Ellipse((state_mom_est[0,0], state_mom_est[1,0]), width, height, angle=angle, alpha=0.3)
    ax.add_artist(ellipse)
    ax.set_title("State Space")
    ax.set_xlabel("State 1")
    ax.set_ylabel("State 2")
    state_color = np.append(state_mom,state_mom_est, axis=1)
    ax.scatter(state_color[0,:], state_color[1,:])
    ax.set_xlim(-50, 0)
    ax.set_ylim(-50, 0)
    plt.draw()
    plt.pause(0.2)

    #propagate through system
    state_mom_est = A @ state_mom
    state_mom = A @ state_mom + noise_state[:, i].reshape(2,1)
    output_mom = H @ state_mom
    cov_state = A @ cov_state @ A.T + B @ cov_nstate @ B.T

    if i == 10 or i == 20:
        K = kalman.kalman(cov_state, cov_nmess, H)
        state_mom_est = state_mom_est + K @ (output_mom - H @ state_mom_est)
        cov_state = cov_state - K @ H @ cov_state

        #calculate ellipse
        a = cov_state[0,0]
        b = cov_state[0,1]
        c = cov_state[1,1]

        width = a + c + np.sqrt((a - c)**2  + 4 * b**2)
        height = a + c - np.sqrt((a - c)**2  + 4 * b**2)
        theta = 0 
        if b == 0 and a >= c:
            theta = 0
        elif b == 0 and a < c:
            theta = np.pi / 2
        else:
            theta = np.arctan2(width/ 2 -a, b)

        angle = np.rad2deg(theta) 

        ellipse = Ellipse((state_mom_est[0,0], state_mom_est[1,0]), width, height, angle=angle, alpha= 0.2, color = "red")
        ax.add_artist(ellipse)
        state_color = np.append(state_mom,state_mom_est, axis=1)
        ax.scatter(state_mom_est[0,:], state_mom_est[1,:])
        plt.draw()
        plt.pause(1)

    i += 1


plt.show() # When all is done
