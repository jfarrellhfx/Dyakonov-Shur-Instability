"""
Jack Farrell, Dept. of Physics, University of  Toronto, 2020

Run the simulation at four different values of v0 and four different values of
eta, all at ratio = 0.001
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import Simulation.annular_v2 as annular

#run for ratio = 0.001
v0 = [0.05,0.10,0.15,0.20]
gamma = 0.1
etas = [0.0001, 0.3333, 0.6666, 0.1]
ratio = 0.001

f = h5py.File("../../Data/ratio-v0.h5","w")
for i in range(len(v0)):
    for j in range(len(etas)):
        n_list, J_list = annular.simulation(ratio, v0[i], etas[j], gamma)
        f.create_dataset("n/v0-{:.3f}/eta-{:.3f}".format(v0[i], etas[j]), data = np.array(n_list))
        f.create_dataset("J/v0-{:.3f}/eta-{:.3f}".format(v0[i], etas[j]), data = np.array(J_list))
f.close()