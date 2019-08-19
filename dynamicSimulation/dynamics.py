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


def getTorsionMatrices(n,G,A,L,r,rho):
    dl = L/n
    m = rho*A*L
    I = m*r**2

    K = np.diag([2.]*n) + np.diag([-1.]*(n-1), k=1) + np.diag([-1.]*(n-1), k=-1)
    K[0,0] = 1
    K[-1,-1] = 1
    K *= G*A/dl

    C = 0.01*K
    # C = 0.0*K

    M = np.diag([2/3.]*n) + np.diag([1/6.]*(n-1), k=1) + np.diag([1/6.]*(n-1), k=-1)
    M[0,0] = 1/3.
    M[-1,-1] = 1/3.
    M *= I/n

    return K, C, M

def getABTorsion(arm, dt):
    # obtain properties from arm
    L = arm.r

    n = arm.state.n
    dl = arm.state.dl

    r = arm.structProps['radius']
    rho = arm.structProps['density']
    delta = arm.structProps['thickness']
    G = arm.structProps['k_rot']

    area = 2*np.pi*r*delta

    # get A and B
    K,C,M = getTorsionMatrices(n,G,area,L,r,rho)
    M_inv = npl.inv(M)


    A = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]),
                   np.hstack([-M_inv@K, -M_inv@C])])
    B_tmp = np.vstack([np.array([1,0]),
                       np.zeros([n-2,2]),
                       np.array([0,1])
                      ])
    B = np.vstack([np.zeros([n,2]),
                   M_inv@B_tmp])

    # discretize dynamics
    t_arr = np.linspace(0,1,10)*dt
    y_arr = np.array([spl.expm(t*A) for t in t_arr])
    tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])

    A_d = spl.expm(A*dt)
    B_d = np.dot(tmp,B)

    return A_d, B_d

def getDeflectionMatrices(n,E,A,L,r,rho):
    dl = L/n
    m = rho*A*L
    dm = m/n

    I = 1./12 * m * (6*r**2 + dl**2)

    K_e = E*I/dl**3 * np.array([
            [   12,    6*dl,   -12,    6*dl],
            [ 6*dl, 4*dl**2, -6*dl, 2*dl**2],
            [  -12,   -6*dl,    12,   -6*dl],
            [ 6*dl, 2*dl**2, -6*dl, 4*dl**2]
            ])

    M_e = rho*A*dl/420 * np.array([
            [    156,    22*dl,     54,   -13*dl],
            [  22*dl,  4*dl**2,  13*dl, -3*dl**2],
            [     54,    13*dl,    156,   -22*dl],
            [ -13*dl, -3*dl**2, -22*dl,  4*dl**2]
            ])

    C_e = 0.01*K_e

    K_tot = np.zeros([2*n,2*n])
    M_tot = np.zeros([2*n,2*n])
    for i in range(n-1):
        K_tot[2*i:2*i+4,2*i:2*i+4] += K_e
        M_tot[2*i:2*i+4,2*i:2*i+4] += M_e

    C_tot = 0.01*K_tot

    return K_tot, C_tot, M_tot

def getABDeflection(arm, dt):
    # obtain properties from arm
    L = arm.r

    n = arm.state.n
    dl = arm.state.dl

    r = arm.structProps['radius']
    rho = arm.structProps['density']
    delta = arm.structProps['thickness']
    E = arm.structProps['k_lat']

    area = 2*np.pi*r*delta

    # get A and B
    K,C,M = getDeflectionMatrices(n,E,area,L,r,rho)
    M_inv = npl.inv(M)


    A = np.vstack([np.hstack([np.zeros([2*n,2*n]), np.eye(2*n)]),
                   np.hstack([-M_inv@K, -M_inv@C])])
    B_tmp = np.vstack([np.array([1,0,0,0]),
                       np.array([0,1,0,0]),
                       np.zeros([2*(n-2),4]),
                       np.array([0,0,1,0]),
                       np.array([0,0,0,1])
                      ])
    B = np.vstack([np.zeros([2*n,4]),
                   M_inv@B_tmp])

    # discretize dynamics
    t_arr = np.linspace(0,1,10)*dt
    y_arr = np.array([spl.expm(t*A) for t in t_arr])
    tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])

    A_d = spl.expm(A*dt)
    B_d = np.dot(tmp,B)

    return A_d, B_d


def simulateTorsion(arm, M1, M2, dt, noise=None):
    """
    Simulate torsional dynamics

    Arguments:
        arm   : Arm object
        M1    : Applied moment at start of boom
        M2    : Applied moment at end of boom
        dt    : Time step interval
        noise : (optional) stdev of noise to inject into dynamics
    Output:
        Updated arm object
    """

    n = arm.state.n

    # get dynamics and control matrices
    A_d,B_d = getABTorsion(arm, dt)

    # setup state
    theta = arm.state.rot_z
    theta_dot = arm.state.rate_z
    X = np.append(theta, theta_dot)

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

def simulateBending(arm, F1, F2, M1, M2, dt, noise=None):
    """
    Simulate torsional dynamics

    Arguments:
        arm   : Arm object
        F1    : Applied force at start of boom
        F2    : Applied force at end of boom
        M1    : Applied moment at start of boom
        M2    : Applied moment at end of boom
        dt    : Time step interval
        noise : (optional) stdev of noise to inject into dynamics
    Output:
        Updated arm object
    """

    n = arm.state.n

    # get dynamics and control matrices
    A_d,B_d = getABDeflection(arm, dt)

    # setup state
    delta = arm.state.def_lat
    ddelta = arm.state.rate_lat
    X = np.append(delta, ddelta)

    # Update step
    u = np.array([F1, M1, F2, M2])
    X_new = np.dot(A_d,X) + np.dot(B_d,u)

    delta_new = X_new[0:2*n]
    ddelta_new = X_new[2*n:]

    if noise:
        delta_new += sigma_theta*np.random.randn(2*n)
        ddleta_new += sigma_dtheta*np.random.randn(2*n)

    arm.state.def_lat = delta_new
    arm.state.rate_lat = ddelta_new

    return arm


if __name__ == "__main__":
    # simple test
    from robotArm import Arm
    import copy

    arm = Arm(1,np.zeros(6), num_fe=10)

    # simulate
    tf = 10.0
    dt = 1e-2

    t_arr = np.arange(0,tf,dt)
    u_arr = np.zeros([2,len(t_arr)])

    # apply an impulse (positive then negative)
    u_arr[0,0] = 1e-6
    u_arr[0,1] = -1e-6

    state_list = []
    state_list.append(copy.copy(arm.state))
    for i in range(len(t_arr)-1):
        u = u_arr[:,i]
        arm = dynamicsStep(arm,u,dt)
        state_list.append(copy.copy(arm.state))

    import simulate
    simulate.plotResults(state_list, np.zeros(len(t_arr)), t_arr)
