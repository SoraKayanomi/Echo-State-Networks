import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from scipy import linalg

a=np.array([1,2,3])
W=np.diag(1-np.power(a,2))
print(W)
