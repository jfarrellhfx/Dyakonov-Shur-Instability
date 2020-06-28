import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate as integrate
h = 1/50
eta = 0.01
v0 = 0.14
gamma = 0.04
n = np.loadtxt("../../Data/vary-ratio/n-ratio-0.001.txt")
J = np.loadtxt("../../Data/vary-ratio/J-ratio-0.001.txt")
v = J / n

R = 0.001
a = 1./R
b = a + 1.
r = np.arange(a, b, h)[:-1]

ra = np.zeros(J.shape)
for i in range(ra.shape[0]):
    ra[i,:] = r

laplacian = []
for i in range(1, J.shape[1] - 1):
    laplacian.append(1/h**2*(v[:, i + 1] - 2 * v[:, i] + v[:, i - 1]) + 1 / ra[:,i] * 1 / h *(v[:, i+1] - v[:, i]))
laplacian = np.array(laplacian).T

conservationpart2 = 1 / h * (n[:,1:] - n[:,:-1])
divee = J ** 2 / n
conservationPart = 1/h * (ra[:,1:] * divee[:,1:] - ra[:,:-1] * divee[:,:-1])

C = integrate.simps(y = v[:,1:-1] * conservationPart[:,1:] + v[:,1:-1]*ra[:,1:-1] * conservationpart2[:,1:], dx = h, axis = 1)

k = 1 / 0.001 * (J[1:] - J[:-1])
K = integrate.simps(y = (v[:-1,:] * ra[:-1,:] * k)[:,1:-1], dx = h, axis = 1)
D = integrate.simps(y = ra[:,1:-1] * v[:,1:-1] * eta * laplacian, dx = h, axis = 1)
MR = integrate.simps(y = v * gamma*(n * v0 * b - ra * J), dx = h, axis = 1)

fig1 = plt.figure()
plt.plot(K, label = "KE")
plt.plot(-D, label = "dissipation")
plt.plot(C, label = "Conservation Law")
plt.plot(-MR, label = "Relaxation")
plt.legend()
plt.xlim(30000, 35000)
plt.show()

fig2 = plt.figure()
plt.plot(K + C[1:])
plt.show()
