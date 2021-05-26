import numpy as np
import matplotlib.pyplot as plt

seriesSize=8000
np.random.seed(2145125)
x=np.random.normal(0,1,(seriesSize,1))
#x=np.random.uniform(0,1,(seriesSize))
#x=np.linspace(0,1000,4000)
#x=np.sin(x)
np.savetxt('iid.txt',x)

plt.figure(10).clear()
plt.plot(x[:500])
plt.title('A sample of data')
plt.show()

