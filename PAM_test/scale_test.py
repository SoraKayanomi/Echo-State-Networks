# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies
import scipy.sparse as ss
import time

# load the data
trainLen = 2000
testLen = 500
initLen = 100
data = np.loadtxt('PAM_length_emulation.txt')
u = data[:, 1]
y = data[:, 2]

# generate the ESN reservoir
inSize = outSize = 1
resSize = 200
a = 1  # leaking rate
np.random.seed(215125)
Win_temp = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    2  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
unimate_rhoW = 0.40370172585965536
W = W*unimate_rhoW/rhoW

optimizeSize = 100
MSE_series = np.zeros(optimizeSize)
scale_series = np.power(10, np.linspace(-2, 2, optimizeSize))
for i in range(optimizeSize):
    scale=scale_series[i]
    Win=Win_temp*scale
    # allocated memory for the design (collected states) matrix
    X = np.zeros((1+inSize+resSize, trainLen-initLen))
    # set the corresponding target matrix directly
    Yt = y[None, initLen:trainLen]

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
    mse = sum(np.square(y[trainLen:trainLen+errorLen] -
                        Y[0, 0:errorLen])) / errorLen
    MSE_series[i] = mse


plt.figure(1).clear()
plt.plot(np.log10(scale_series), MSE_series, 'g')
plt.xlabel(r'$log_{10}(scale)$')
plt.ylabel('MSE')

output = np.vstack((scale_series, MSE_series))
np.savetxt('scale_test.txt', output.T)

print('minMSE:')
print(np.min(MSE_series))
print('occurs at scale=')
print(scale_series[np.argmin(MSE_series)])


'''
# plot some signals
plt.figure(1).clear()
plt.plot(y[trainLen+1:trainLen+testLen+1], 'g')
plt.plot(Y.T, 'b')
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])
'''

plt.show()
