import numpy as np
import matplotlib.pyplot as plt
import Simulation.annular_v2 as annular

n_list, J_list = annular.simulation(3.200, 0.14, 0.01, 0.04)

np.savetxt("../../Data/vary-ratio/n-ratio-3.200.txt", np.array(n_list))
np.savetxt("../../Data/vary-ratio/J-ratio-3.200.txt", np.array(J_list))