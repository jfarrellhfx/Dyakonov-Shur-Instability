"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Makes plots illustrating the DS instability that does not exist in a
rectangular geometry can exist in the corbino geometry.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size=12)

n_list = []
n_list.append(np.loadtxt("../../Data/check-existence/n-ratio-0.001.txt"))
n_list.append(np.loadtxt("../../Data/check-existence/n-ratio-2.500.txt"))
t = np.linspace(0.0, 35.0, int(35.0/0.001) + 1)

fig1, axes1 = plt.subplots(1,2,sharex=True, sharey = True, figsize = (8,3))
axes1[0].plot(t, n_list[0][:,-1], color = "black")
axes1[0].set_xlabel("$v_st/L$")
axes1[1].set_xlabel("$v_st/L$")
axes1[0].set_ylabel("$n(R_2,t)/n_0$")
axes1[0].set_title("$\mathcal{R} = 0.001$")
axes1[1].set_title("$\mathcal{R} = 2.500$")
axes1[1].plot(t, n_list[1][:,-1], color = "black")
axes1[0].grid()
axes1[1].grid()
plt.tight_layout()
plt.savefig("../../Figures/check-existence/check-existence.pdf")
plt.show()