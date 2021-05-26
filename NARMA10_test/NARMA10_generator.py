import numpy as np
import matplotlib.pyplot as plt

size=8000
np.random.seed(21489214)
x=(np.random.rand(size,1) * 0.5)
y=np.zeros((size,1))
for i in range(1,size):
    k=i
    if i>=11:
        k=10
    y[i]=0.3*y[i-1]
    sum=0
    for j in range(k):
        sum+=y[i-1-j]
    y[i]+=0.05*y[i-1]*sum
    y[i]+=1.5*x[i-10]*x[i-1]+0.1
data=np.vstack((x.T,y.T))
np.savetxt('NARMA10.txt',data.T)

plt.figure(10).clear()
plt.plot(y[:500],'b')
plt.plot(x[:500],'y')
plt.title('A sample of data')
plt.show()

