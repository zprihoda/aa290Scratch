"""
-------------------
Reference Material:
-------------------

Torsional:
Reference equations on torsional and bending vibrations for beam elements:
    http://www.wind.civil.aau.dk/lecture/7sem/notes/Lecture9.pdf
Duke Lectures on StructDyn
    http://people.duke.edu/~hpgavin/cee541/
Universite Libre de Bruxelles Course Notes:
    https://scmero.ulb.ac.be/Teaching/Courses/MECA-H-303/MECA-H-303-Lectures.pdf
Numerial Solution to Wave equation:
    http://www-users.math.umn.edu/~olver/num_/lnp.pdf

Bending / General FE methods:
http://www.solid.iei.liu.se/Education/TMHL08/Lectures/Lecture__8.pdf
http://homepages.cae.wisc.edu/~suresh/ME964Website/M964Notes/Notes/introfem.pdf
DYNAMIC FINITE ELEMENT METHODS - Lecture notes for SD2450 - Biomechanics and Neuronics

Finite Element for Vibrations:
http://www1.aucegypt.edu/faculty/mharafa/MENG%20475/Finite%20Element%20Vibration%20Analysis%20Fall%202010.pdf

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
    Cleanup inputs and outputs

Implement ability to fix one of the ends
    Need to look into how fixing an end affects our dynamics model
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import structuralProperties as structProp

sigma_theta = 1e-3
sigma_dtheta = 1e-3

def dynamicsStep(arm, u, dt, noise=False):
    """
    Main Dynamics Function called in simulation

    Inputs:
        arm: current robot arm object
        u    : control commands
            u = [M1, ... , M6, F_r]
            M1-M6 = Command moments for each rotational motor
            F_r = Force in radial direction
        dt  : time step
        noise: boolean indicating whether to inject dynamic noise
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
    arm = simulateTorsion(arm, M1, M2, dt, noise)

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

def getABMatrices(n, k, c, I, dt):
    C = getRelationMatrix(n)
    A = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]),
                   np.hstack([k/I*C, c/I*C])
                   ])
    B = np.vstack([np.zeros([n,2]),
                   np.array([1/I,0]),
                   np.zeros([n-2,2]),
                   np.array([0,1/I])
                   ])

    # discretize dynamics
    A_d = spl.expm(A*dt)

    t_arr = np.linspace(0,1,10)*dt
    y_arr = np.array([spl.expm(t*A) for t in t_arr])

    tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])
    B_d = np.dot(tmp,B)

    return A_d, B_d

def simulateTorsion(arm, M1, M2, dt, noise):
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
    r = arm.structProps['radius']
    rho = arm.structProps['density']
    delta = arm.structProps['thickness']

    I_fe = structProp.getBoomInertia(dl, r, delta, rho)[2,2]     # moment of inertia
    k = arm.structProps['k_rot']
    c = arm.structProps['c_rot']

    theta = arm.state.rot_z
    theta_dot = arm.state.rate_z
    X = np.append(theta, theta_dot)

    # get dynamics and control matrices
    A_d, B_d = getABMatrices(n, k, c, I_fe, dt)

    # Update step
    u = np.array([M1,M2])
    X_new = np.dot(A_d,X) + np.dot(B_d,u)

    theta_new = X_new[0:n]
    theta_dot_new = X_new[n:]

    if noise:
        theta_new += sigma_theta*np.random.randn(n)
        theta_dot_new += sigma_dtheta*np.random.randn(n)

    arm.state.rot_z = theta_new
    arm.state.rate_z = theta_dot_new

    return arm

def simulateBending(arm, M1, M2, dt):
    # TODO: Implement and test
    raise NotImplementedError('Bending not Implemented yet')

    # obtain properties from arm
    n = arm.state.n
    dl = arm.state.dl
    I = structProp.getBoomInertia(dl)[0,0]     # moment of inertia
    k = arm.structProps['k_lat']
    c = arm.structProps['c_lat']

    u = arm.state.lat_x
    u_dot = arm.state.lat_dx
    X = np.append(u, u_dot)


if __name__ == "__main__":
    # simple test
    from robotArm import Arm
    import copy

    arm = Arm(1,np.zeros(6))

    # simulate
    tf = 1.0
    dt = 1e-3

    t_arr = np.arange(0,tf,dt)
    u_arr = np.zeros([2,len(t_arr)])

    # apply an impulse (positive then negative)
    u_arr[0,0] = 1e-6
    u_arr[0,1] = -1e-6

    state_list = []
    for i,t in enumerate(t_arr):
        u = u_arr[:,i]
        arm = dynamicsStep(arm,u,dt)
        state_list.append(copy.copy(arm.state))

    import simulate
    simulate.plotResults(state_list,t_arr)
