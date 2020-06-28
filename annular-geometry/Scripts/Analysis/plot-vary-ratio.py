"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020
Script to plot the DS instability of viscous electrons in a Corbino geometry
at various different values of R1 relative to the channel length L
"""


import numpy as np
import matplotlib.pyplot as plt
import os

# Matplotlib Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size=12)

# Setup Storage
ratios = []
n_list = []
J_list = []
t = np.linspace(0.0, 35.0, int(35.0/0.001) + 1)

# Load Data
for filename in os.listdir("../../Data/vary-ratio"):
    if filename.startswith("n"):
        ratios.append(float(filename[8:13]))
        n_list.append(np.loadtxt("../../Data/vary-ratio/" + filename))
        print(filename)
    elif filename.startswith("J"):
        J_list.append(np.loadtxt("../../Data/vary-ratio/" + filename))
        print(filename)
#Plot
fig1, axes1 = plt.subplots(2, 4, figsize = (8, 4), sharex = True, sharey = True)
i = 0
for axlist in axes1:
    for ax in axlist:
        ax.plot(t, n_list[i][:,-1], color = "black")
        ax.set_title("$\mathcal{R} =$ " + "{:.3f}".format(ratios[i]))
        i += 1
        ax.grid()
for ax in [axlist[0] for axlist in axes1]:
    ax.set_ylabel("$n(R_2,t)/n_0$")
for ax in axes1[1]:
    ax.set_xlabel("$v_st/L$")
plt.tight_layout()
plt.savefig("../../Figures/vary-ratio/vary-ratio.pdf")
plt.show()
