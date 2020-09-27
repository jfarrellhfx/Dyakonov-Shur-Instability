"""
Jack Farrell, Dept. of Physics, University of Toronto, 2020
Script to study the Dyakonov - Shur Instability of viscous electrons in
graphene in an annular domain.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import DS_config as config

# Basic Parameters ----------------------------------------------------------------------------------------------------|
k = config.k
h = config.h
T = config.T
delta = config.delta

imageLog = config.imageLog # if True, shows some figure
saveFigures = False #if True, saves pdfs of all figures


# Helper Functions ----------------------------------------------------------------------------------------------------|
def minmod(a, b):
    """
    One of the slope-limiting options used in the high resolution
    correction.  This one is known to be pretty diffusive.
    """
    if np.abs(a) < np.abs(b) and a * b > 0:
        return a
    elif np.abs(b) < np.abs(a) and a * b > 0:
        return b
    else:
        return 0


def maxmod(a, b):
    """
    Helper function for superbee limiter
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
    i.e., numpy arrays with 2 components.  This is needed to solve the
    "Riemmann Problem"
    """

    nleft, Jleft = uleft
    nright, Jright =  uright

    if nleft <= 0:
        print("nLeft is a problem")
    if nright <= 0:
        print("nright is a problem")

    jump_in = uright - uleft
    v_in = (nleft ** 0.5 * Jleft / nleft + nright ** 0.5 * Jleft / nright) / (nleft ** 0.5 + nright ** 0.5)
    # Want to solve for "alpha1, alpha2", the coefficients of each eigenvector
    # in the expansion of the jump.  We do that here!
    alpha1_in = (jump_in[0] * (v_in + 1) - jump_in[1]) / 2
    alpha2_in = (-jump_in[0] * (v_in - 1) + jump_in[1]) / 2
    return alpha1_in, alpha2_in


def f(u_current):
    """
    flux term in equation.  In our case, it is a vector [J, J**2/n + n].
    """
    n_in, J_in = u_current
    return np.array([J_in, J_in ** 2 / n_in + n_in])


def harten(eigenvalue):
    if np.abs(eigenvalue) <= delta:
        return (eigenvalue**2 + delta**2) / 2 / delta
    else:
        return np.abs(eigenvalue)


def CFL(q):
    v_max = np.max(np.abs(q))
    return h ** 2 / v_max




def flux(uL, uR, UL, UR):
    """
    Finite volume methods approximate the average of the solution on a bunch of
    cells.  The averages are updated by the "flux" through the boundaries of the
    cell.  This function computes the flux at the boundary between uL and uR.
    Doing so means solving something called a "Riemann Problem" at each cell
    boundary.  I use Roe's Approximate Riemann Solver for isothermal flow,
    which is well documented. Because of the high -
    resolution method used, the values at one previous cell, UL and UR are also
    needed.
    """

    nL, JL = uL
    nR, JR = uR


    v = (nL ** 0.5 * JL / nL + nR ** 0.5 * JR / nR) / (nL ** 0.5 + nR ** 0.5)

    # Hard-code eigenvectors and eigenvalues
    r1 = [1, v - 1]


    w1 = v - 1.
    r2 = [1, v + 1]
    w2 = v + 1.

    #Expand jump in state vectors in eigenvectors of Roe Matrix.
    alpha1, alpha2 = eigenExpand(uL, uR)

    # Entropy Fix
    # (There's a special case called a transonic rarefaction in which the Roe
    # solver fails.  This is because the Roe solver can give solutions that
    # violate an important principle called the "entropy condition".  But since
    # this is only wrong in the case of a transonic rarefaction, we can apply
    # a "entropy fix" that deals with that case.

    """
    u1L = uL
    u1R = u1L + alpha1 * np.array(r1)

    u2L = uL + alpha1 * np.array(r1)
    u2R = u2L + alpha2 * np.array(r2)


    y1L = u1L[1] / u1L[0] - 1
    y1R = u1R[1] / u1R[0] - 1

    y2L = u2L[1] / u2L[0] + 1
    y2R = u2R[1] / u2R[0] + 1
    if y1R != y1L and y2R != y2L:
        B1 = (y1R - w1) / (y1R - y1L)
        B2 = (y2R - w2) / (y2R - y2L)


    if y1L < 0 and 0 <= y1R:
        w1L = B1 * y1L
    else:
        w1L = min(w1, 0)
    if y2L < 0 and 0 <= y1R:
        w2L = B2 * y2L
    else:
        w2L = min(w2, 0)"""

    w1L = min(w1, 0)
    w2L = min(w2, 0)
    #w1L = 1 / 2 * (w1 - harten(w1))
    #w2L = 1 / 2 * (w2 - harten(w2))

    # Get the gobunov flux
    fG = f(uL) + alpha1 * w1L * np.array(r1) + alpha2 * w2L * np.array(r2)

    # For next part, need the "j - 1" value of the alpha1, alpha2 expansion
    # coefficients.  I give these new variable names by switching the case
    # of everything.
    NL, jL = UL
    NR, jR = UR
    NL, NR = np.abs(NL), np.abs(NR)
    V = (NL ** 0.5 * jL / NL + NR ** 0.5 * jR / NR) / (NL ** 0.5 + NR ** 0.5)

    R1 = [1, V - 1]
    R2 = [1, V + 1]
    Alpha1, Alpha2 = eigenExpand(UL, UR)

    # Slope Limiter
    sigma1 = 1 / h * np.array(
        [slopeLimiter(alpha1 * r1[0], Alpha1 * R1[0]), slopeLimiter(alpha1 * r1[1], Alpha1 * R1[1])])
    sigma2 = 1 / h * np.array(
        [slopeLimiter(alpha2 * r2[0], Alpha2 * R2[0]), slopeLimiter(alpha2 * r2[1], Alpha2 * R2[1])])


    # The high resolution correction to the flux:
    additionalFlux = 1 / 2 * (w1 * (np.sign(k * w1 / h) - k * w1 / h) * h * sigma1 + w2 * (np.sign(k * w2/ h) - k * w2 / h) * h * sigma2)

    F = fG + additionalFlux

    return F


# *** SELECT WHICH SLOPE LIMITER ***
slopeLimiter = minmod


# Run Simulation ------------------------------------------------------------------------------------------------------|
def simulation(ratio = 0.0, v0 = 0.0, eta = 0.0, gamma =0.0):
    global k

    """
    Run the Simulation

    Parameters
    ----------
    ratio - ratio of (R2 - R1) / R1
    v0, eta, gamma - dimensionless simulation parameters

    Returns:
    ----------
    n_array, J_array - arrays of simulation results, time on axis 0, space on
    axis 1
    """

    # *** PARAMETERS FOR POLAR GEOMETRY ***
    # We keep the separation constant and use the ratio defined below to
    # calculate new values for rMin and rMax
    sep = 1.0
    rMin = sep / ratio
    R1 = rMin
    R2 = sep + R1
    r = np.arange(R1, R2, h)

    # like to do computations at midpoints for finite volume methods
    rMid = r + h / 2.

    # We want to input v_a, the steady state velocity at the inner radius.  But
    # we are fixing the momentum on the outer radius.  So we convert the inner
    # radius value to the outer radius value here.
    v0 = R1 / R2 * v0


    def relaxationStep(u):
        """
        Convenience function to calculate the source terms given the dataframe
        u whose 2 columns are the fields n and J.
        """
        return np.array([
            (-u[:, 1] / rMid),
             gamma * (R2/rMid*v0 * u[:,0] - u[:, 1]) - u[:, 0] * (R2** 2) / (rMid**3) * v0 ** 2  + (-u[:, 1] ** 2 / u[:, 0] / rMid) - 1 * eta * (u[:, 1]/u[:,0]) / rMid**2
        ]).T



    #The set of times at which the simulation will be run



    #Set up initial conditions as little perturbations on the steady state.

    perturb = 0.2
    n0 = 1. + perturb * np.sin(np.pi*(rMid-R1))
    J0 = v0 * (1.0 + 0 * np.cos(np.pi*(rMid - R1)/2)) * R2 / rMid

    u = np.vstack((n0, J0)).T

    #Storage
    J_list = []
    n_list = []

    J_list.append(np.copy(u[:,1]))
    n_list.append(np.copy(u[:,0]))

    #Initialize Counter
    N = 0

    # Extend "r" past the computational domain a bit, needed in calculating
    # boundary conditions
    rBC = np.concatenate((np.array([rMid[0] - 2 * h, rMid[0] - h]), rMid, np.array([rMid[-1] + h, rMid[-1] + 2 * h])))
    time = 0
    t = []
    t.append(0)

    switched = False
    k0 = k
    # Main loop!
    while time < T:

        # Pick a time step based on the maximum velocity.  If it gets too high,
        # need to drastically reduce time step or it becomes unstable.  But note
        # that we will still always ***save*** data
        if np.max(u[:,1] / u[:,0]) > 1.5:
            switched = True

        if switched == True:
            k = 0.0001
        else:
            k = 0.001

        # First part of the Strang Splitting - integrate the relaxation term
        # up to dt / 2
        u1 = u + k / 4 * relaxationStep(u)
        u = u + k / 2 * relaxationStep(u1)

        # Set Boundary Conditions
        uLeft = np.array([[1, u[2,1]*rMid[2]/rBC[0]], [1, u[1,1]*rMid[1]/rBC[1]]])
        uRight = np.array([[u[-1, 0], v0], [u[-1, 0], v0]])
        uBC = np.vstack((uLeft, u, uRight))
        q = uBC[:,1]/uBC[:,0]

        U = np.copy(u)

        # Iterate through *physical* domain
        for j in range(2, uBC.shape[0] - 2):

            # Call the flux() function at the left and right boundary of each
            # cell to be able to update each element.
            FMinus = flux(uBC[j - 1], uBC[j], uBC[j - 2], uBC[j - 1])
            FPlus = flux(uBC[j], uBC[j + 1], uBC[j - 1], uBC[j])

            #Calculate dissipative contribution
            dissipative = k * np.array([0, eta *(1/h**2 * (q[j + 1] - 2 * q[j] + q[j - 1]) + 1 / rBC[j] / 2 / h * (q[j + 1] - q[j - 1]))])
            # Update each element of physical domain
            U[j - 2] = u[j-2] - k / h * (FPlus - FMinus) + dissipative
        u = np.copy(U)

        # Second step of Strang Splitting, same integration
        u1 = u + k / 4 * relaxationStep(u)
        u = u + k / 2 * relaxationStep(u1)

        # Store Data:
        # if we have switched to the lower timestep, save data only every 10
        # iterations.  Otherwise, save data every iteration.
        if switched and N % 10 == 0:
            J_list.append(np.copy(u[:, 1]))
            n_list.append(np.copy(u[:, 0]))
        elif not switched:
            J_list.append(np.copy(u[:, 1]))
            n_list.append(np.copy(u[:, 0]))

        # Log Progress
        if N % 1000 == 0:
            print("Time is {:.3f} out of {:.3f} - - - - Iteration {}".format(time, T, N))
            print("Time Step: {:.5f}".format(k))
        if N % 2000 == 0 and imageLog:
            fig, axes = plt.subplots(1, 2, figsize = (6,3))
            ax1, ax2 = axes
            ax1.plot([l[-1] for l in n_list], color = "black")
            ax1.set_title("n(L,t)")
            ax2.plot(np.array(rMid * J_list[-1]) / rMid[0], color = "Blue")
            ax2.plot(np.array(rMid * n_list[-1]) / rMid[0], color= "Purple")
            ax2.set_title("Snapshots")
            ax1.grid()
            ax2.grid()
            if N == 50000:
                plt.savefig("brightside.pdf")
            plt.show()

        N += 1
        time += k



    return np.array(n_list), np.array(J_list)
































