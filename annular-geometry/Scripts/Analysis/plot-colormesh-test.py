import numpy as np
import matplotlib.pyplot as plt
import os

data1 = np.loadtxt("../../Data/vary-ratio/n-ratio-0.001.txt")
data2 = np.loadtxt("../../Data/vary-ratio/n-ratio-2.000.txt")

plt.pcolormesh(data2)
plt.colorbar()
plt.show()
plt.plot(data2[35000,:])
plt.show()