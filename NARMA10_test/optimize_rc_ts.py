# Optimize the ridge-regression coefficient
# and the training length

#from matplotlib import scale
import numpy as np
#import matplotlib.pyplot as plt
#from numpy import random
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies
import matplotlib.pyplot as plt

# load the data
data = np.loadtxt('NARMA10.txt')
u = data[:, 0]
y = data[:, 1]

# generate the ESN reservoir
inSize = 1
outSize = 1
resSize = 200
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

trainlenOptimizeSize = 20
regOptimizeSize = 14
trainlen_series = np.arange(
    20, (20*(trainlenOptimizeSize+1)), 20).reshape(trainlenOptimizeSize, 1)
reg_series = np.power(np.sqrt(10.), np.arange(-8, regOptimizeSize-8)
                      ).reshape(regOptimizeSize, 1)
# determination coefficient
NMSE_series = np.zeros((regOptimizeSize,trainlenOptimizeSize))
for i in range(regOptimizeSize):
    # define the ridge-regression coefficient
    reg = reg_series[i]
    for j in range(trainlenOptimizeSize):
        trainLen = trainlen_series[j][0]
        testLen = 1000
        initLen = 200

        print('i:', end=' ')
        print(i, end=' ')
        print('j:', end=' ')
        print(j)

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
        nmse = sum(np.square(y[trainLen+initLen+1:trainLen+initLen+errorLen+1] -
                             Y[0, 0:errorLen])) / np.linalg.norm(y[trainLen+initLen+1:trainLen+initLen+errorLen+1], 2)
        NMSE_series[i, j] = nmse
        '''
        # plot some signals
        plt.figure(1).clear()
        plt.plot(y[trainLen+initLen+1:trainLen+initLen+testLen+1], 'g')
        plt.plot(Y.T, 'b')
        plt.title('Target and generated signals $y(n)$ starting at $n=0$')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        plt.show()
        '''

NMSE_series = np.vstack((trainlen_series.T, NMSE_series))
NMSE_series = np.hstack(
    (np.vstack((np.zeros((1, 1)), reg_series)), NMSE_series))
np.savetxt('optimize_rc_ts.txt', NMSE_series)
