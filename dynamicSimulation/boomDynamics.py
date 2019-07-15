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

TODO: How to handle state when we are extending
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
n = 10
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

def simulateTorsion(M1,M2,x):
    """
    Simulate torsional dynamics

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
    for i in range(t_steps):
        dx = simulateTorsion(M1[i],M2[i],x)

        x = x + dx*dt
        x_arr[:,i+1] = x

    # plot stuff
    for i in range(n):
        pl.plot(x_arr[i,:])
    pl.legend(range(n))
    pl.title('Torsional Finite-Element Model')
    pl.show()
