import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies
import scipy.sparse as ss
import time

# load the data
trainLen = 1000
testLen = 1000
delay = 0
data = np.loadtxt('iid.txt')

# plot some of it
#plt.figure(10).clear()
#plt.plot(data[:1000])
#plt.title('A sample of data')

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
inputScale = 1
a = 1  # leaking rate
np.random.seed(114514)
Win = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    inputScale  # Win as Nx*(Nu+1) matrix (+1 means the bias)
W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
unimate_rhoW = 1.25
W *= unimate_rhoW / rhoW

# allocated memory for the design (collected states) matrix
X = np.zeros((1+inSize+resSize, trainLen))
# set the corresponding target matrix directly
Yt = data[None, 0:trainLen]

# run the reservoir with the data and collect X
x = np.zeros((resSize, 1))
for t in range(trainLen+delay):
    u = data[t]
    x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
    if t >= delay:
        # record the x in each step in training part
        X[:, t-delay] = np.vstack((1, u, x))[:, 0].reshape(resSize+inSize+1)

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
u = data[trainLen+delay]
for t in range(testLen):
    x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(Wout, np.vstack((1, u, x)))
    Y[:, t] = y
    # generative mode:
    u = y
    # this would be a predictive mode:
    #u = data[trainLen+t+1]

# compute MSE for the first errorLen time steps
errorLen = 50
mse = sum(np.square(data[trainLen+1:trainLen+errorLen+1] -
                    Y[0, 0:errorLen])) / errorLen
print('MSE = ' + str(mse))

# plot some signals
plt.figure(1).clear()
plt.plot(data[trainLen+1:trainLen+testLen+1], 'g')
plt.plot(Y.T, 'b')
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])


plt.show()