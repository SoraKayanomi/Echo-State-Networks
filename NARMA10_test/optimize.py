#from matplotlib import scale
import numpy as np
#import matplotlib.pyplot as plt
#from numpy import random
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies

# load the data
trainLen = 2000
testLen = 500
initLen = 100
delay = 20

data = np.loadtxt('NARMA10.txt')
u = data[:, 0]
y = data[:, 1]

# generate the ESN reservoir
inSize = 1
outSize = delay
resSize = 100
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

rhoOptimizeSize = 50
scaleOptimizeSize = 50
rhoW_series = np.linspace(0.05, 0.05*rhoOptimizeSize, rhoOptimizeSize).reshape(rhoOptimizeSize, 1)
scale_series = np.linspace(
    0.05, 0.05*scaleOptimizeSize, scaleOptimizeSize).reshape(scaleOptimizeSize, 1)
# determination coefficient
MSE_series = np.zeros((rhoOptimizeSize, scaleOptimizeSize))
for i in range(rhoOptimizeSize):
    # define the ultimate spectral radius
    W = W_temp*(rhoW_series[i])
    for j in range(scaleOptimizeSize):
        # define the ultimate scale
        Win = Win_temp*(scale_series[j])

        # allocated memory for the design (collected states) matrix
        X = np.zeros((1+inSize+resSize, trainLen-initLen))
        # set the corresponding target matrix directly
        Yt = y[None, initLen+1:trainLen+1]

        # run the reservoir with the data and collect X
        x = np.zeros((resSize, 1))
        for t in range(trainLen):
            u_current = u[t]
            x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
            if t >= initLen:
                # record the x in each step in training part
                X[:, t-initLen] = np.vstack((1, u_current, x)
                                            )[:, 0].reshape(resSize+inSize+1)

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
            u_current = u[trainLen+t]
            x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
            y_current = np.dot(Wout, np.vstack((1, u_current, x)))
            Y[:, t] = y_current

        # compute MSE for the first errorLen time steps
        errorLen = testLen
        mse = sum(np.square(y[trainLen+1:trainLen+errorLen+1] -
                            Y[0, 0:errorLen])) / errorLen
        MSE_series[i,j]=mse

MSE_series = np.vstack((scale_series.T, MSE_series))
MSE_series = np.hstack((np.vstack((np.zeros((1, 1)), rhoW_series)), MSE_series))
np.savetxt('optimize.txt', MSE_series)
