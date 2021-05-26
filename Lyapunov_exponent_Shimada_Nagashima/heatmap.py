import matplotlib.pyplot as plt
import numpy as np
origin = np.loadtxt('optimize_s50.txt')

data = origin

#imshow
plt.figure(1).clear()
plt.axis('on')  # scale
plt.imshow(data, origin='lower')
plt.colorbar(shrink=0.8).set_label('MLLE')  # shrink 用于调整colorbar大小
#plt.xticks(np.array([0,4,9,14,19,24,29,34,39]),np.array([0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]))
plt.xticks(np.array([0,3,6,9,12,15,18]),np.array([0.1,0.4,0.7,1.0,1.3,1.6,1.9]))
plt.yticks(np.array([0,4,9,14,19,24,29,34,39]),np.array([0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]))
plt.xlabel('input scale')
plt.ylabel('spectral radius')

#countour
plt.figure(2).clear()
plt.contourf(data)
plt.colorbar(shrink=0.8).set_label('MLLE')  # shrink 用于调整colorbar大小
#plt.xticks(np.array([0,4,9,14,19,24,29,34,39]),np.array([0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]))
plt.xticks(np.array([0,3,6,9,12,15,18]),np.array([0.1,0.4,0.7,1.0,1.3,1.6,1.9]))
plt.yticks(np.array([0,4,9,14,19,24,29,34,39]),np.array([0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]))
plt.xlabel('input scale')
plt.ylabel('spectral radius')

plt.show()
print(0)