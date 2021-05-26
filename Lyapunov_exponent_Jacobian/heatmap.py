import matplotlib.pyplot as plt
import numpy as np
origin = np.loadtxt('optimize.txt')

data = origin

#imshow
plt.figure(1).clear()
plt.axis('on')  # scale
plt.imshow(data, cmap='Reds', origin='lower')
plt.colorbar(shrink=0.8).set_label('Memory Capacity')  # shrink 用于调整colorbar大小
#plt.xticks(np.array([0,4,9,14,19,24,29,34,39,44,49]),np.array([0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]))
#plt.yticks(np.array([0,4,9,14,19,24,29,34,39]),np.array([0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]))
plt.xlabel('input scale')
plt.ylabel('spectral radius')

#countour
plt.figure(2).clear()
plt.contourf(data)
plt.colorbar(shrink=0.8).set_label('Memory Capacity')  # shrink 用于调整colorbar大小
#plt.xticks(np.array([0,4,9,14,19,24,29,34,39,44,49]),np.array([0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]))
#plt.yticks(np.array([0,4,9,14,19,24,29,34,39]),np.array([0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]))
plt.xlabel('input scale')
plt.ylabel('spectral radius')

plt.show()
print(0)