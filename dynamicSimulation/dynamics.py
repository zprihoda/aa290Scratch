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

----
TODO
----
How to handle state when we are extending
    ie. while extending how does our state (x) change
    We may be able to linearly interpolate to find new state:
        x_new_pos = L/n * np.arange... (?)

Implement dynamics for bending

Implement Dynamics class
    Calculate instance variables once:
        eg. Relation Matrix
    Store structural properties
    Get rid of these awful global varaibles
    Cleanup inputs and outputs

Implement ability to fix one of the ends
    Need to look into how fixing an end affects our dynamics model
"""

import numpy as np
import structuralProperties as structProp


def dynamicsStep(arm, u, dt):
    """
    Main Dynamics Function called in simulation

    Inputs:
        arm: current robot arm object
        u    : control commands
            u = [M1, ... , M6, F_r]
            M1-M6 = Command moments for each rotational motor
            F_r = Force in radial direction
        dt  : time step
    """

    # Determine Command Forces
    M1_cmd = u[0]
    M2_cmd = u[1]

    # TODO: Determine external forces
    M1_ext = 0
    M2_ext = 0

    M1 = M1_cmd + M1_ext
    M2 = M2_cmd + M2_ext

    # Torsion simulation
    arm = simulateTorsion(arm, M1, M2, dt)


    # Bending Simulation
    # TODO: Implement

    # Extension Simulation
    # TODO: Implement
    # F_cmd = u[-1]
    # F_ext = 0   # TODO
    # Fz = F_cmd + F_ext
    # arm.state = simulateExtension(arm.state, Fz)    # NOTE: simulate extension also remaps all other states to the new finite element model

    return arm



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

def simulateTorsion(arm, M1, M2, dt):
    """
    Simulate torsional dynamics

    Arguments:
        x: state [theta_1 ... theta_n, theta_dot_1 ... theta_dot_n]
        M1: Applied moment at start of boom
        M2: Applied moment at end of boom
    Output:
        x_new: updated state
    """

    # obtain properties from arm
    n = arm.state.n
    dl = arm.state.dl
    I = structProp.getBoomInertia(dl)[2,2]     # moment of inertia
    k = arm.structProps['k_rot']
    c = arm.structProps['c_rot']

    theta = arm.state.rot_z
    theta_dot = arm.state.rate_z
    X = np.append(theta, theta_dot)

    # solve struct Dynamics problem
    C = getRelationMatrix(n)
    A = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]),
                   np.hstack([k/I*C, c/I*C])
                   ])

    u = np.hstack([np.zeros(n),M1/I,np.zeros(n-2),M2/I])
    dX = np.dot(A,X) + u

    X_new = X + dX*dt
    theta_new = X_new[0:n]
    theta_dot_new = X_new[n:]

    arm.state.rot_z = theta_new
    arm.state.rate_z = theta_dot_new

    return arm
