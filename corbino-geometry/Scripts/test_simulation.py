"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020

Reproduce the results of Mendl et al. as a limit L / a -> 0 of the full
Corbino situation, as a test
"""

import annular as annular
import numpy as np
import matplotlib.pyplot as plt

t_list, n_list, J_list = annular.simulation(ratio = 0.001, v0 = 0.5, eta = 0.03, gamma = 0.1)
np.savetxt("test3.txt", n_list)