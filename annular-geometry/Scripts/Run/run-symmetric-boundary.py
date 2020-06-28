"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Demonstrating that the oscillation with symmetric boundary conditions has
double the frequency of the asymmetric boundary conditions.  But haven't yet
been able to observe the endpoint of the instability.
"""

import Simulation.symmetricBC as symmetricBC
import numpy as np

params = (0.14, 0.01, 0.04)
n_list, J_list = symmetricBC.simulation(0.1, *params)
np.savetxt("../../Data/symmetric-boundary/n-decay-demo.txt", np.array(n_list))
np.savetxt("../../Data/symmetric-boundary/J-decay-demo.txt", np.array(J_list))