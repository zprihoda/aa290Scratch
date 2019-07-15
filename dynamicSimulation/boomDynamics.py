"""
-------------------
Reference Material:
-------------------
Reference equations on torsional and bending vibrations for beam elements:
    http://www.wind.civil.aau.dk/lecture/7sem/notes/Lecture9.pdf
Duke Lectures on StructDyn
    http://people.duke.edu/~hpgavin/cee541/
Universite Libre de Bruxelles Course Notes:
    https://scmero.ulb.ac.be/Teaching/Courses/MECA-H-303/MECA-H-303-Lectures.pdf
Numerial Solution to Wave equation:
    http://www-users.math.umn.edu/~olver/num_/lnp.pdf
Damped Wave Equation Source:
    http://www.math.psu.edu/tseng/class/Math251/Notes-PDE%20pt4.pdf


-----
TODO:
-----

How to handle state when the boom is extending
    ie. while extending how does our state (x) change
    We may be able to linearly interpolate to find new state:
        x_new_pos = L/n * np.arange
"""

import numpy as np
from matplotlib import pyplot as pl

import structuralProperties as structProp


k = 1
c = 0.001
L = 1.
n = 5
I = structProp.getBoomInertia(L/n)[2,2]

dt = 1e-4

def getRelationMatrix(n):
    """
    return the relation matrix for our finite element mode
    For a 3 element model:
        C =[[-1, 1, 0],
            [ 1,-2, 1],
            [ 0, 1,-1]]
    """

    tmp = -2*np.ones(n)
    tmp[0] = -1
    tmp[-1] = -1
    C = np.diag(tmp)

    C += np.diag([1]*(n-1),k=1)
    C += np.diag([1]*(n-1),k=-1)

    return C

def simulateTorsion(x,M1,M2):
    """
    Simulate torsional dynamics using finite-element method

    Arguments:
        x: state [theta_1 ... theta_n, theta_dot_1 ... theta_dot_n]
        M1: Applied moment at start of boom
        M2: Applied moment at end of boom
    Output:
        dx: change of state with respect to time
    """

    C = getRelationMatrix(n)

    A = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]),
                   np.hstack([k/I*C, c/I*C])
                   ])

    u = np.hstack([np.zeros(n),M1/I,np.zeros(n-2),M2/I])
    dx = np.dot(A,x) + u
    return dx


def simulateTorsion2(X, X_prev,M1,M2):
    """
    Simulate torsional dynamics using numerical approximation to wave equation

    Arguments:
        Arguments:
        X       : current state [theta_1 ... theta_n, theta_dot_1 ... theta_dot_n]
        X_prev  : previous state
        M1      : Applied moment at start of boom
        M2      : Applied moment at end of boom
    Output:
        dX      : change of state with respect to time
    """

    # dt = dt     # from script namespace
    dx = L/n
    j = I/dx
    gamma = -100   # damping coeff
    c_wave = np.sqrt(50e9/structProp.BOOM_DENSITY)
    c_wave = 100
    sigma = c_wave*dt/dx

    A = np.diag([2*(1-sigma**2 + 0.5*gamma*dt)]*n) + np.diag([sigma**2]*(n-1),k=1) + np.diag([sigma**2]*(n-1),k=-1)
    M = np.zeros(n); M[0]=M1; M[-1]=M2

    X_new = np.dot(A,X) - (1+gamma*dt)*X_prev + dt**2/j*M
    return X_new


if __name__ == "__main__":

    x = np.zeros(2*n)

    # simulate
    t_steps = 1000
    M1 = np.zeros(t_steps)
    M2 = np.zeros(t_steps)
    M1[0] = 1e-3
    M1[1] = -M1[0]

    x_arr = np.zeros([len(x),t_steps+1])
    x_arr[:,0] = x

    x2 = np.zeros(n)
    x2_arr = np.zeros([len(x2),t_steps+1])
    x2_arr[:,0] = x2

    for i in range(t_steps):
        dx = simulateTorsion(x,M1[i],M2[i])
        x = x + dx*dt
        x_arr[:,i+1] = x

        # Note on first iteration x_arr[:,-1] is filled with zeros (should probably be more explicit here)
        x2 = simulateTorsion2(x2,x2_arr[:,i-1], M1[i], M2[i])
        print x2
        x2_arr[:,i+1] = x2

    # plot stuff
    # pl.figure()
    # for i in range(n):
        # pl.plot(x_arr[i,:])
    # pl.legend(range(n))
    # pl.title('Torsional Finite-Element Model')
    # pl.show()

    pl.figure()
    for i in range(n):
        pl.plot(x2_arr[i,:])
    pl.legend(range(n))
    pl.title('Torsional Wave Eq Model')
    pl.show()

