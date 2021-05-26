#from matplotlib import scale
import numpy as np
#import matplotlib.pyplot as plt
#from numpy import random
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies

# load the data
trainLen = 500
testLen = 500
initLen = 100
delay = 20

data = np.loadtxt('iid.txt')

# generate the ESN reservoir
inSize = 1
outSize = delay
resSize = 25
a = 1  # leaking rate
np.random.seed(124512)
Win_temp = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    1  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W_temp = W/rhoW

rhoOptimizeSize = 40
scaleOptimizeSize = 50
rhoW_series = np.linspace(0.05, 0.05*rhoOptimizeSize,
                          rhoOptimizeSize).reshape(rhoOptimizeSize, 1)
scale_series = np.linspace(
    0.01, 0.01*scaleOptimizeSize, scaleOptimizeSize).reshape(scaleOptimizeSize, 1)
# determination coefficient
MC = np.zeros((rhoOptimizeSize, scaleOptimizeSize))
for i in range(rhoOptimizeSize):
    # define the ultimate spectral radius
    W = W_temp*(rhoW_series[i])
    for j in range(scaleOptimizeSize):
        # define the ultimate scale
        Win = Win_temp*(scale_series[j])

        # allocated memory for the design (collected states) matrix
        X = np.zeros((1+inSize+resSize, trainLen))
        # set the corresponding target matrix directly
        Yt = np.zeros((delay, trainLen))
        for k in range(trainLen):
            if(k <= delay):
                temp = data[0:k]
                temp = temp[::-1]
                Yt[0:k, k] = temp
            else:
                temp = data[k-delay:k]
                temp = temp[::-1]
                Yt[0:k, k] = temp

        # run the reservoir with the data and collect X
        x = np.zeros((resSize, 1))
        for t in range(trainLen):
            u_current = data[t]
            x = (1-a)*x + a * \
                np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
            # record the x in each step in training part
            X[:, t] = np.vstack((1, u_current, x))[
                :, 0].reshape(resSize+inSize+1)

        # train the output by ridge regression
        reg = 1e-8  # regularization coefficient
        # direct equations from texts:
        #X_T = X.T
        # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        #    reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        Wout = linalg.solve(np.dot(X, X.T) + reg*np.eye(1+inSize+resSize),
                            np.dot(X, Yt.T)).T

        # run the trained ESN in a generative mode. no need to initialize here,
        # because x is initialized with training data and we continue from there.
        Y = np.zeros((outSize, testLen))
        for t in range(testLen):
            u_current = data[trainLen+t]
            x = (1-a)*x + a * \
                np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
            y_current = np.dot(Wout, np.vstack((1, u_current, x)))
            Y[:, t] = y_current.reshape(delay)

        # compute the MC in each rho
        for k in range(delay):
            x = np.vstack(
                (data[trainLen:trainLen+testLen-k-1].T, Y[k, k+1:].T))
            MC[i, j] += np.power(np.corrcoef(x)[0, 1], 2)

MC = np.vstack((scale_series.T, MC))
MC = np.hstack((np.vstack((np.zeros((1, 1)), rhoW_series)), MC))
# print(MC)
np.savetxt('optimize.txt', MC)
'''
plt.figure(1).clear()
plt.plot(scale_series,MC[0])
plt.show()
'''
