import numpy as np
import matplotlib.pyplot as plt
import time as tm

seriesSize = 8000
np.random.seed(int(tm.time()))
# x=np.random.normal(0,1,(seriesSize,1))

# according to the Verstraeten,-2010 Memory versus Non-Linearity in Reservoirs
x = np.random.uniform(-0.8, 0.8, (seriesSize))
np.savetxt('iid.txt', x)

plt.figure(10).clear()
plt.plot(x[:500])
plt.title('A sample of data')
plt.show()
