# -*- coding: utf-8 -*-
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
from https://mantas.info/code/simple_esn/
(c) 2012-2020 Mantas Lukoševičius
Distributed under MIT license https://opensource.org/licenses/MIT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# numpy.linalg is also an option for even fewer dependencies
import scipy.sparse as ss
import time

# load the data
trainLen = 2000
testLen = 2000
initLen = 100
data = np.loadtxt('iid.txt')

# plot some of it
plt.figure(10).clear()
plt.plot(data[:1000])
plt.title('A sample of data')

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
a = 1  # leaking rate
np.random.seed(114514)
Win = (np.random.rand(resSize, 1+inSize) - 0.5) * \
    1  # Win as Nx*(Nu+1) matrix (+1 means the bias)
# W = np.random.rand(resSize, resSize) - 0.5  # W as Nx*Nx matrix
# sparese the matrix
dense = 10
num_ele = resSize*dense  # connecting num to every activation in reservoir

#W = np.zeros((resSize,resSize))
arr_a = [np.random.randint(0, resSize) for _ in range(num_ele)]
arr_b = [np.random.randint(0, resSize) for _ in range(
    num_ele-resSize)] + [i for i in range(resSize)]  # 保证每一列都有值，不会出现全零列
arr_c = [np.random.rand()-0.5 for _ in range(num_ele)]
rows, cols, v = np.array(arr_a), np.array(arr_b), np.array(arr_c)
sparseX = ss.coo_matrix((v, (rows, cols)))
W = sparseX.todense()

# normalizing and setting spectral radius (correct, slow):
# time_start=time.time() 

print('Computing spectral radius...')
rhoW = max(abs(linalg.eig(W)[0]))
print('done.')
unimate_rhoW = 1.25
W *= unimate_rhoW / rhoW

# time_end=time.time() 
# print('totally cost',time_end-time_start)

# allocated memory for the design (collected states) matrix
X = np.zeros((1+inSize+resSize, trainLen-initLen))
# set the corresponding target matrix directly
Yt = data[None, initLen+1:trainLen+1]

# run the reservoir with the data and collect X
x = np.zeros((resSize, 1))
for t in range(trainLen):
    u = data[t]
    x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
    if t >= initLen:
        # record the x in each step in training part
        X[:, t-initLen] = np.vstack((1, u, x))[:, 0].reshape(resSize+inSize+1)

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
u = data[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(Wout, np.vstack((1, u, x)))
    Y[:, t] = y
    # generative mode:
    u = y
    # this would be a predictive mode:
    #u = data[trainLen+t+1]

# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum(np.square(data[trainLen+1:trainLen+errorLen+1] -
                    Y[0, 0:errorLen])) / errorLen
print('MSE = ' + str(mse))


# plot some signals
plt.figure(1).clear()
plt.plot(data[trainLen+1:trainLen+testLen+1], 'g')
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
