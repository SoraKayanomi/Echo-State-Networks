import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies
import scipy.sparse as ss
import time

# load the data
trainLen = 2000
testLen = 1000
initLen = 100
data = np.loadtxt('PAM_length_emulation.txt')
u = data[2000:, 1]
y = data[2000:, 2]

# generate the ESN reservoir
inSize = outSize = 1
resSize = 200
a = 1  # leaking rate
np.random.seed(215125)
Win_temp = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    2*4.229242874389499  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
unimate_rhoW = 0.5
W = W*unimate_rhoW/rhoW

optimizeSize = 1
MSE_series = np.zeros(optimizeSize)
scale_series = np.array([0.8])
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
    reg = 1e-4  # regularization coefficient
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

print('MES=')
print(MSE_series[0])

# plot some signals
plt.figure(1).clear()
plt.plot(y[trainLen:trainLen+testLen], 'g')
plt.plot(Y.T, 'b')
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])

'''
plt.figure(2).clear()
plt.plot(X[0:20, 0:200].T)
plt.title(r'Some reservoir activations $\mathbf{x}(n)$')

plt.figure(3).clear()
plt.bar(np.arange(1+inSize+resSize), Wout[0].T)
plt.title(r'Output weights $\mathbf{W}^{out}$')
'''
plt.show()
