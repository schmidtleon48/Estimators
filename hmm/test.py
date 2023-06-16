import numpy as np
import matplotlib.pyplot as plt

mean = np.array([0, 0])
cov = np.array([[2,np.sqrt(2)],[np.sqrt(2),1]])

a = np.random.multivariate_normal(mean,cov, 10)

print(cov)
print(a)
print(a[:,0]/ a[:,1])
print(np.sqrt(2))