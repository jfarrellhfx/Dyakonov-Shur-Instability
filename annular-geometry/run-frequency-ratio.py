import numpy as np
import matplotlib.pyplot as plt
import os
import annular

# *** Simulation Parameters ***
k = annular.k
h = annular.h
T = annular.T

number = 10

eta = 0.01
gamma = 0.14

ratios = np.linspace(0.001, 3.0, number)
v0s = np.linspace(0.1, 0.5, number)

results = np.zeros((number, number))

for i in range(number):
    print("Working on ratio {}".format(ratios[i]))
    for j in range(number):
        print("Working on v0 {}".format(v0s[j]))
        n_array, J_array = annular.simulation(ratios[i], v0s[j], eta, gamma)
        results[i, j] = annular.frequency(n_array[:,-1], k)
        plt.pcolormesh(v0s, ratios, results)
        plt.xlabel("v0")
        plt.ylabel("ratio")
        plt.colorbar()
        plt.show()

np.savetxt("Data/frequency-ratio/frequencyRatio1.txt", results)