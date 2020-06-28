"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020
Simualte the DS instability of viscous electrons in an annular domain at various
values of R1 relative to the channel length L.
"""

import numpy as np
import os
import Simulation.annular as annular

ratios =[0.001, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
params = (0.14, 0.01, 0.04)


for R in ratios:
    n_list, J_list = annular.simulation(R, *params)
    np.savetxt("../../Data/vary-ratio/n-ratio-{:.3f}.txt".format(R), np.array(n_list))
    np.savetxt("../../Data/vary-ratio/J-ratio-{:.3f}.txt".format(R), np.array(J_list))
