"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

At the parameters used, in the rectangular geometry, we don't get any
instability, but in an annular geometry, if it's 'annular enough' we can see
an instability.  Here, ratio = 0.001 has no instability (it decays) but
ratio = 2.500 does have an instability.
"""

import numpy as np
import Simulation.annular as annular

params = (0.04,0.01,0.1)

n_list, J_list = annular.simulation(0.001, *params)
np.savetxt("../../Data/check-existence/n-ratio-0.001.txt", np.array(n_list))

n_list,J_list =annular.simulation(2.500, *params)
np.savetxt("../../Data/check-existence/n-ratio-2.500.txt", np.array(n_list))