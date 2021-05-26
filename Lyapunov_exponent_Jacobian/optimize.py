import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# load the data
trainLen = 1000
initLen = 100

data = np.loadtxt('iid.txt')

# generate the ESN reservoir
inSize = 1
resSize = 100
a = 1  # leaking rate
np.random.seed(1366436)
Win_temp = (np.random.rand(resSize, inSize) - 0.5) * \
    1  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix
b = 1  # bias in activation function

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W_temp = W/rhoW

rhoOptimizeSize = 10
scaleOptimizeSize = 10
rhoW_series = np.linspace(0.2, 0.2*rhoOptimizeSize, rhoOptimizeSize).reshape(rhoOptimizeSize, 1)
scale_series = np.linspace(
    0.2, 0.2*scaleOptimizeSize, scaleOptimizeSize).reshape(scaleOptimizeSize, 1)
#rhoW_series = np.array([0.1,0.5,1,1.5])
#scale_series = np.array([0.2,0.6,1,1.4])
# Maximal Local Lyapunov Exponent
MLLE = np.zeros((rhoOptimizeSize, scaleOptimizeSize))

# determination coefficient
MC = np.zeros((rhoOptimizeSize, scaleOptimizeSize))
for i in range(rhoOptimizeSize):
    # define the ultimate spectral radius
    W = W_temp*(rhoW_series[i])
    for j in range(scaleOptimizeSize):
        # define the ultimate scale
        Win = Win_temp*(scale_series[j])

        # ith eigenvalue of the Jacobian at a time k
        Li = np.zeros((trainLen-initLen, resSize))

        # run the reservoir with the data and collect X
        x = np.zeros((resSize, 1))
        for t in range(trainLen):
            u_current = data[t]
            x = (1-a)*x + a * \
                np.tanh(np.dot(Win, u_current) + np.dot(W, x)+b)
            # record the x in each step in training part
            if(t >= initLen):
                J = np.diag(1-np.power(x, 2))*W  # Jacobian at a this timestep
                Li[initLen-t] = linalg.eig(J)[0]  # eigenvector of the Jacobian

        # print(Li[0])
        LLE = np.ones((resSize))  # whole spectrum of Lyapunov exponents
        for ii in range(resSize):
            for k in range(trainLen-initLen):
                LLE[ii] *= np.power(abs(Li[k, ii]),
                                    float(1)/(trainLen-initLen))
        LLE = np.log(LLE)

        MLLE[i, j] = max(LLE)

np.savetxt('optimize.txt', MLLE)
print(0)