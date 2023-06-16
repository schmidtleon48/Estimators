import numpy
import matplotlib.pyplot as plt

# Generate polynomial
nSteps = 121
coeff = [-50, 70, -16, 1]
sigmaNoise = 50
sigmaPrior = 100
xMax = 10

ts = numpy.linspace(0,xMax,nSteps)
deltaT = ts[1] - ts[0]
nPoly = len(coeff)
A = numpy.array([ts**i for i in range(nPoly)])
y_polynomial = coeff @ A

# Noise
numpy.random.seed(1)
noise = sigmaNoise*numpy.random.randn(nSteps)

# Add noise to the signal
y = y_polynomial + noise

# Prepare Kalman estimation
D = numpy.zeros((nPoly,nPoly))
D[(numpy.arange(nPoly-1), numpy.arange(nPoly-1)+1)] = 1
Dt = D*deltaT
F = numpy.identity(nPoly) + Dt + Dt @ Dt/2 + Dt @ Dt @ Dt/6
H = numpy.zeros((1,nPoly))
H[0,0] = 1

# Initialize Kalman estimation
x = numpy.zeros(nPoly)
P = sigmaPrior**2 * numpy.identity(nPoly)

# Start Kalman iteration
yEst = []
ySigma = []
for i in range(len(y)):
    # Propagate
    if i > 0:
        x = F @ x
        P = F @ P @ F.T

    # Estimate
    K = P @ H.T @ numpy.linalg.inv(H @ P @ H.T + sigmaNoise**2)
    x = x + K @ (y[i] - H @ x)
    P = (numpy.identity(nPoly) - K @ H) @ P
    ySigma.append(P[0,0])
    yEst.append(x[0])

ySigma = numpy.sqrt(ySigma)

# Select prior state for GLS estimation

p = numpy.ones(nPoly)/sigmaPrior
p[2] *= 2
p[3] *= 6

# Iterative GLS estimation

yLSI = []
for i in range(nSteps):
    nm = min(nPoly, i+1)
    RN = numpy.zeros((i+1,i+1))
    RN[numpy.diag_indices(RN.shape[0])] = 1/sigmaNoise**2
    RI = numpy.zeros((nm,nm))
    RI[numpy.diag_indices(RI.shape[0])] = p[:nm]**2
    Ai = A[:nm,:i+1]
    cLS = numpy.linalg.inv(Ai @ RN @ Ai.T + RI) @ Ai @ RN @ y[:i+1]
    yLSI.append(cLS @ Ai[:,i])


# Plot
plt.figure(figsize=(6,3.2))
#plt.fill_between(ts,y_polynomial-ySigma, y_polynomial+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
#plt.plot(ts,y,'.-', color='C1', markersize=4, linewidth=0.4, alpha=0.6, label='Polynom + Rauschen')
plt.plot(ts,y_polynomial,'C2', label='Polynom 3. Grades')
plt.plot(ts,yEst,'c-', color='#2f97ff',  label='Kalman-Schätzung')
plt.plot(ts,yLS,'k-', alpha=0.25, label='Kleinste-Quadrate-Schätzung (ohne a-priori)')
plt.plot(ts,yLSI,'C3--' ,label='Kleinste-Quadrate-Schätzung (mit a-priori)')
plt.xlabel('Zeit')
plt.legend(loc=4)
plt.tight_layout()
plt.savefig('Kalman_Polynom_vs_GLS.svg')
plt.savefig('Kalman_Polynom_vs_GLS.png')
#plt.show()