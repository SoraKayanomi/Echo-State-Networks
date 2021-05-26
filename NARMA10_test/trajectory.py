# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies
import scipy.sparse as ss
import time

# load the data
data = np.loadtxt('NARMA10.txt')
u = data[:, 0]
y = data[:, 1]

trainLen = 2000
testLen = 500
initLen = 200

# generate the ESN reservoir
inSize = 1
outSize = 1
resSize = 50
a = 1  # leaking rate
np.random.seed(41521)
Win = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    1  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W = W / rhoW * 1.05     # ultimate spectral radius

# allocated memory for the design (collected states) matrix
X = np.zeros((1+inSize+resSize, trainLen))
# set the corresponding target matrix directly
Yt = y[None, initLen+1:trainLen+initLen+1]

# run the reservoir with the data and collect X
x = np.zeros((resSize, 1))
for t in range(trainLen+initLen):
    u_current = u[t]
    x = (1-a)*x + a * \
        np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
    if t >= initLen:
        # record the x in each step in training part
        X[:, t-initLen] = np.vstack((1, u_current, x)
                                    )[:, 0].reshape(resSize+inSize+1)

# train the output by ridge regression
reg = 1e-8
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
    u_current = u[trainLen+initLen+t]
    x = (1-a)*x + a * \
        np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
    y_current = np.dot(Wout, np.vstack((1, u_current, x)))
    Y[:, t] = y_current

# compute MSE for the first errorLen time steps
errorLen = testLen
mse = sum(np.square(y[trainLen+initLen+1:trainLen+initLen+errorLen+1] -
                        Y[0, 0:errorLen])) / errorLen

# plot some signals
plt.figure(1).clear()
plt.plot(y[trainLen+initLen+1:trainLen+initLen+testLen+1], 'g')
plt.plot(Y.T, 'b')
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])


plt.figure(2).clear()
plt.plot(X[0:20, 0:200].T)
plt.title(r'Some reservoir activations $\mathbf{x}(n)$')

plt.figure(3).clear()
plt.bar(np.arange(1+inSize+resSize), Wout[0].T)
plt.title(r'Output weights $\mathbf{W}^{out}$')

plt.show()