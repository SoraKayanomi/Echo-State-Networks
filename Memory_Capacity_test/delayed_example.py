# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies

# load the data
trainLen = 5000
testLen = 500
initLen = 100
delay = 20

data = np.loadtxt('iid.txt')

# generate the ESN reservoir
inSize = 1
outSize = delay
resSize = 200
a = 1  # leaking rate
np.random.seed(124512)
Win = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    4/100  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
W_temp = W/rhoW

optimizeSize = 1
MSE_series = np.zeros(optimizeSize)
rhoW_series = np.linspace(0.5,0.5, optimizeSize)
for i in range(optimizeSize):
    unimate_rhoW = rhoW_series[i]
    W = W_temp*unimate_rhoW
    # allocated memory for the design (collected states) matrix
    X = np.zeros((1+inSize+resSize, trainLen))
    # set the corresponding target matrix directly
    Yt=np.zeros((delay,trainLen))
    for j in range(trainLen):
        if(j<=delay):
            temp=data[0:j]
            temp=temp[::-1]
            Yt[0:j,j]=temp
        else:
            temp=data[j-delay:j]
            temp=temp[::-1]
            Yt[0:j,j]=temp

    # run the reservoir with the data and collect X
    x = np.zeros((resSize, 1))
    for t in range(trainLen):
        u_current = data[t]
        x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
        # record the x in each step in training part
        X[:, t] = np.vstack((1, u_current, x))[:, 0].reshape(resSize+inSize+1)

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
        x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u_current))) + np.dot(W, x))
        y_current = np.dot(Wout, np.vstack((1, u_current, x)))
        Y[:, t] = y_current.reshape(delay)

C=np.zeros((delay,1))#determination coefficient
for i in range(delay):
    x=np.vstack((data[trainLen:trainLen+testLen-i-1].T,Y[i,i+1:].T))
    C[i]=np.power(np.corrcoef(x)[0,1],2)

print(np.sum(C))

plt.figure(1).clear()
plt.plot(C)
plt.xlabel("delay(timestep)")
plt.ylabel("determination coefficient")

plt.figure(2).clear()
plt.plot(data[trainLen:trainLen+50],linewidth=1)
plt.plot(Y[1,:50],linewidth=0.5)
plt.plot(Y[3,:50],linewidth=0.5)
plt.plot(Y[5,:50],linewidth=0.5)
#plt.plot(Y[8,:50],linewidth=0.5)
plt.legend(['origin','delay=2', 'delay=4', 'delay=6'])

plt.show()
