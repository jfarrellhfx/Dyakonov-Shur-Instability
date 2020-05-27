""" Jack Farrell, Dept. of Physics, University of Toronto, 2020
Script to study the Dyakonov - Shur Instability of viscous electrons in
graphene in an annular domain.
"""

import numpy as np
import matplotlib.pyplot as plt

# Basic Parameters ----------------------------------------------------------------------------------------------------|
k = 0.001
h = 1 / 50.
T = 50.

imageLog = True # if True, shows some figure
saveFigures = False #if True, saves pdfs of all figures


# Functions -----------------------------------------------------------------------------------------------------------|
def minmod(a, b):
    """ One of the slope-limiting options used in the high resolution
    correction.  This one is known to be pretty diffusive.
    """
    if np.abs(a) < np.abs(b) and a * b > 0:
        return a
    elif np.abs(b) < np.abs(a) and a * b > 0:
        return b
    else:
        return 0


def maxmod(a, b):
    """ Helper function for superbee limiter
    """
    if np.abs(a) > np.abs(b) and a * b > 0:
        return a
    elif np.abs(b) > np.abs(a) and a * b > 0:
        return b
    else:
        return 0


def superbee(a, b):
    """
    A less diffusive slope-limiter than minmod
    """
    s1 = minmod(b, 2 * a)
    s2 = minmod(2 * b, a)
    return maxmod(s1, s2)


def eigenExpand(uleft, uright):
    """
    Expands jump in state vector from cell to cell in terms of
    the eigenfunctions of the Roe matrix 'A' which is hard
    coded in.  uL and uR should be two-component state vectors,
    i.e., numpy arrays with 2 components.
    """
    nleft, Jleft = uleft
    nright, Jright = uright
    jump_in = uright - uleft
    v_in = (nleft ** 0.5 * Jleft / nleft + nright ** 0.5 * Jleft / nright) / (nleft ** 0.5 + nright ** 0.5)
    # Want to solve for "alpha1, alpha2", the coefficients of each eigenvector
    # in the expansion of the jump.  We do that here!
    alpha1_in = (jump_in[0] * (v_in + 1) - jump_in[1]) / 2
    alpha2_in = (-jump_in[0] * (v_in - 1) + jump_in[1]) / 2

    return alpha1_in, alpha2_in


def f(u_current):
    """ flux term in equation.  In our case, the it is a vector [J, J**2/n + n].
    Just a convenience function that does this calcul
    """
    n_in, J_in = u_current
    return np.array([J_in, J_in ** 2 / n_in + n_in])


def flux(uL, uR, UL, UR):
    """ Finite volume methods approximate the average of the solution on a bunch of
    cells.  The averages are updated by the "flux" through the boundaries of the
    cell.  This function computes the flux at the boundary between uL and uR.
    Because of the high resolution method used, the values at one previous cell,
    UL and UR are also needed.
    """
    nL, JL = uL
    nR, JR = uR

    # rho averaged velocity
    v = (nL ** 0.5 * JL / nL + nR ** 0.5 * JR / nR) / (nL ** 0.5 + nR ** 0.5)
    # Hard-code eigenvectors and eigenvalues (from [3])
    r1 = [1, v - 1]
    w1 = v - 1
    r2 = [1, v + 1]
    w2 = v + 1
    alpha1, alpha2 = eigenExpand(uL, uR)

    # Get the gobunov flux
    fG = f(uL) + min(w1, 0) * alpha1 * np.array(r1) + min(w2, 0) * alpha2 * np.array(r2)

    # For next part, need the "j - 1" value of the alpha1, alpha2 expansion
    # coefficients.  I give these new variable names by switching the case
    # of everything
    NL, jL = UL
    NR, jR = UR
    V = (NL ** 0.5 * jL / NL + NR ** 0.5 * jR / NR) / (NL ** 0.5 + NR ** 0.5)
    R1 = [1, V - 1]
    R2 = [1, V + 1]
    Alpha1, Alpha2 = eigenExpand(UL, UR)

    # Here's the slope limiting thing
    sigma1 = 1 / h * np.array(
        [slopeLimiter(alpha1 * r1[0], Alpha1 * R1[0]), slopeLimiter(alpha1 * r1[1], Alpha1 * R1[1])])
    sigma2 = 1 / h * np.array(
        [slopeLimiter(alpha2 * r2[0], Alpha2 * R2[0]), slopeLimiter(alpha2 * r2[1], Alpha2 * R2[1])])

    # The high resolution correction to the flux:
    additionalFlux = 1 / 2 * (w1 * (np.sign(k * w1 / h) - k * w1 / h) * h * sigma1 + w2 * (
                np.sign(k * w2 / h) - k * w2 / h) * h * sigma2)

    F = fG + additionalFlux

    return F


# *** SELECT WHICH SLOPE LIMITER ***
slopeLimiter = minmod


# Run Simulation ------------------------------------------------------------------------------------------------------|

# *** PARAMETERS FOR POLAR GEOMETRY ***
# We keep the separation constant and use the ratio defined below to calculate
# new values for rMin and rMax
ratio = 2.13 # Ratio of ring separation to inner radius.
sep = 1.0
rMin = sep / ratio
r = np.arange(rMin, rMin + sep, h)
tau = np.arange(0., T, k)

#like to do computations at midpoints for finite volume methods
rMid = r[:-1] + h / 2.

# *** SIMULATION PARAMETERS ***
v0 = 0.14 # [] dimensionless velocity
eta = 0.01 # [] dimensionless viscosity
gamma = 0.04 # [] dimensionless momentum relaxation rate

n0 = 1. + 0.1 * np.sin((rMid-rMin)*np.pi)
J0 = v0 * (1. + 0.2 * np.cos(np.pi*(rMid - rMin) / 2))

u = np.vstack((n0, J0)).T

#Storage
J_list = []
n_list = []

J_list.append(np.copy(u[:,1]))
n_list.append(np.copy(u[:,0]))
N = 0

for t in np.arange(0, T, k):
    # First part of the Strang Splitting - integrate the relaxation term up to
    # dt / 2
    u = np.copy(np.array([
        u[:,0] + (-u[:,1] / rMid) * k/2,
        u[:,1] + k/2 * gamma * (u[:,0]*v0 - u[:,1]) + k/2*(-u[:,1]**2/u[:,0]/rMid)
    ]).T)

    # Now impose boundary conditions on left and right!
    uLeft = np.array([[1., u[1][1]], [1., u[1][1]]])
    uRight = np.array([[2 * u[-1][0] - u[-2][0], v0], [u[-1, 0], v0]])
    uBC = np.vstack((uLeft, u, uRight))
    U = np.copy(u)

    n = uBC[:, 0]  # just useful
    J = uBC[:, 1]
    q = J / n

    # Iterate through *physical* domain
    for j in range(2, uBC.shape[0] - 2):
        # Call the flux() function at the left and right boundary of each cell
        FMinus = flux(uBC[j - 1], uBC[j], uBC[j - 2], uBC[j - 1])
        FPlus = flux(uBC[j], uBC[j + 1], uBC[j - 1], uBC[j])

        # Approximate the dissipative term by a finite differences quotient (2nd order)
        dissipative = k * np.array([
            0,
            eta * 1 / h ** 2 * (q[j + 1] - 2 * q[j] + q[j - 1]) + eta * 1/rMid[j - 2] * 1/h * (q[j] - q[j - 1])
        ])

        # Update each element of physical domain
        U[j - 2] = u[j - 2] - k / h * (FPlus - FMinus) + dissipative

    # Second step of Strang Splitting, same integration
    U = np.copy(np.array([
        U[:,0] + (-U[:,1] / rMid) * k/2,
        U[:,1] + k/2*gamma * (U[:,0]*v0 - U[:,1]) + k/2*(-U[:,1]**2/U[:,0]/rMid)
    ]).T)

    # send to the storage lists
    J_list.append(np.copy(U[:, 1]))
    n_list.append(np.copy(U[:, 0]))
    u = np.copy(U)

    # Log Progress
    if N % 500 == 0:
        print("Time is {:.3f} out of {:.3f} - - - - Iteration {}".format(t, T, N))
    if N % 1000 == 0 and imageLog:
        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes
        ax1.plot([l[-1] for l in n_list])
        ax1.set_title("n(L,t)")
        ax2.plot(J_list[-1])
        ax2.set_title("J")
        plt.show()
    N += 1

# Save some data ------------------------------------------------------------------------------------------------------|

np.savetxt("Data/dsAnnular-n-{:.3f}.txt".format(ratio), np.array(n_list))
np.savetxt("Data/dsAnnular-J-{:.3f}.txt".format(ratio), np.array(J_list))
















































