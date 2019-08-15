
"""
1: https://en.wikibooks.org/wiki/Strength_of_Materials/Torsion
    Torsional Strenght Equation:
        phi = TL/(Gj)
"""

import numpy as np
import structuralProperties as structProp
import dynamics

import cvxpy as cp

from robotArm import Arm

K_ROT = Arm.structProps['k_rot']
C_ROT = Arm.structProps['c_rot']
BOOM_THICKNESS = Arm.structProps['thickness']
BOOM_RADIUS = Arm.structProps['radius']
BOOM_DENSITY = Arm.structProps['density']

def controlStep(y, waypoint, mode, dt):
    """
    arguments:
        arm : current arm object
        waypoint: desired state
        mode: fsm mode
    outputs:
        u   : control vector
    """

    # TODO: include other controllers
    # TODO: build total u vector

    if mode == 'none':
        u =  [0,0]
    elif mode == 'damping':
        y_rot = y['rot_z']

        r = y['r']
        I = structProp.getBoomInertia(r)[2,2]

        ddtheta = torsionDampingControl(y_rot)
        M = I*ddtheta
        u = M   # assume we are commanding the moment

    elif mode == 'mpc':
        u = mpcController(y,dt)

    return u

def torsionDampingControl(y):
    """
    Torsional Damping Controller

    Inputs:
        y: measurements [theta_start, theta_end, w_start, w_end]
    Outputs:
        u: ddtheta (d^2(theta)/dt^2
    """
    g = 10.0

    w1 = y['rate_z'][0]
    w2 = y['rate_z'][-1]

    u1 = 0
    u2 = -g*(w2-w1)
    u = np.array([u1,u2])
    return u


def mpcController(y,dt):
    n = len(y['rot_z'])
    dl = y['r']/n
    I_fe = structProp.getBoomInertia(dl, BOOM_RADIUS, BOOM_THICKNESS, BOOM_DENSITY)[2,2]

    A,B = dynamics.getABMatrices(n, K_ROT, C_ROT, I_fe, dt)

    X0 = np.append(y['rot_z'], y['rate_z'])
    X_des = np.zeros(2*n)
    X_des[0:n] = np.pi
    T = 50  # time horizon

    # solve cvx problem
    z = cp.Variable([2*n,T])
    u = cp.Variable([2,T-1])

    g_d = 1     # damping
    g_c = 1     # control
    g_f = 2     # final

    u_max = 1e-3    # max control
    T_max = 1e-1       # max torsion

    J_d = cp.sum(cp.abs(z[n,:]-z[-1,:]))        # damping
    J_c = cp.sum(cp.abs(u))                     # control
    J_f = cp.norm(z[:,-1]-X_des)                # final
    obj = cp.Minimize(g_d*J_d + g_c*J_c + g_f*J_f)

    constr = []
    constr += [z[:,0] == X0]                    # initial condition
    constr += [z[:,1:] == A@z[:,:-1] + B@u]     # FE dynamics
    constr += [cp.abs(u) <= u_max]              # control limits
    constr += [cp.abs(z[0:n-1,:] - z[1:n,:])  <= T_max*dl/K_ROT] #torsional strength [1]

    prob = cp.Problem(obj, constr)
    try:
        result = prob.solve()
        u_opt = u.value[:,0]
    except cp.error.SolverError:
        print("Warning: cvx solver failed")
        # TODO: should be able to extract the best control from
        #   cvxpy when it fails. OR, just use a solver that doesn't fail
        #   e.g. MOSEK
        u_opt = [0,0]

    return u_opt
