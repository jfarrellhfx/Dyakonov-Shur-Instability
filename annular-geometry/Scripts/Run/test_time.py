import numpy as np
import Simulation.annular_v2 as annular
import time
start_time = time.now()
n_list, J_list = annular.simulation(0.001, 0.14, 0.01, 0.04)
end_time = time.now()

print("Duration", start_time - end_time)