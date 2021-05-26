import numpy as np
import os as os

np.random.seed(982140)
a=np.random.normal(size=(5,1))
b=np.random.normal(size=(5,1))
x=np.vstack((a.T,b.T))
#print(x)
print(np.corrcoef(x))
