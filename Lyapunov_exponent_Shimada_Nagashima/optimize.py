import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os as os
import time as tm

# load the data
trainLen = 2000  # should be sufficiently large
initLen = 1000  # should be sufficiently large

data = np.loadtxt('iid.txt')

# generate the ESN reservoir
inSize = 1
resSize = 50
np.random.seed(324123)
Win_temp = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    1  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W_temp = W/rhoW

rhoOptimizeSize = 40
scaleOptimizeSize = 20
rhoW_series = np.linspace(0.05, 0.05*rhoOptimizeSize,
                          rhoOptimizeSize).reshape(rhoOptimizeSize, 1)
scale_series = np.linspace(
    0.1, 0.1*scaleOptimizeSize, scaleOptimizeSize).reshape(scaleOptimizeSize, 1)
#rhoW_series = np.array([1.05])
#scale_series = np.array([1])
# Maximal Local Lyapunov Exponent
MLLE = np.zeros((rhoOptimizeSize, scaleOptimizeSize))
np.random.seed(int(tm.time()))
epsilonInit = np.random.uniform(-1e-8, 1e-8, (resSize, 1))

for i in range(rhoOptimizeSize):
    # define the ultimate spectral radius
    W = W_temp*(rhoW_series[i])
    for j in range(scaleOptimizeSize):
        # define the ultimate scale
        Win = Win_temp*(scale_series[j])

        print('i:', end=' ')
        print(i, end=' ')
        print('j:', end=' ')
        print(j)

        LE = 0
        x = np.zeros((resSize, 1))
        x_pert = np.zeros((resSize, 1))
        u_current = 0

        # compute the MLLE based on Shimada-Nagashima method
        # washoff phase at init cause x begins at 0
        for t in range(trainLen):
            u_current = data[t]
            if(t >= initLen):
                x_pert = x + epsilonInit
                x_pert = np.tanh(np.dot(Win, np.vstack(
                    (1, u_current))) + np.dot(W, x_pert))
            x = np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
            if(t >= initLen):
                LE += np.log(np.linalg.norm((x_pert-x), 2) /
                             np.linalg.norm(epsilonInit, 2))
        MLLE[i, j] = LE/(trainLen-initLen)


# print(MLLE[-1])
np.savetxt('optimize.txt', MLLE)
'''
plt.figure(1).clear()
plt.plot(rhoW_series,MLLE)
plt.show()
'''

# os.system('shutdown -s -f -t 60')
