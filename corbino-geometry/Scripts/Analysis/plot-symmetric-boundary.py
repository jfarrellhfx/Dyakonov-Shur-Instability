"""
Jack Farrell., Dept. of Physics, University of Toronto, 2020

Plot the decaying solution with symmetric boundary conditions.  The oscillations
have double the frequency of the asymmetric boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Matplotlib Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size=12)

data = np.loadtxt("../../Data/symmetric-boundary/n-decay-demo.txt")
t = np.linspace(0.0, 35.0, int(35.0/0.001) + 1)

fig1 = plt.figure(figsize = (6, 3))
plt.plot(t, data[:,-1], color = "Black")
plt.grid()
plt.xlabel("$v_st/L$")
plt.ylabel("$J(R_2,t)/n_0$")
plt.tight_layout()
plt.savefig("../../Figures/symmetric-boundary/symmetric-boundary.pdf")
plt.show()
