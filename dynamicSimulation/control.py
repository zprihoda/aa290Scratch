import numpy as np
import structuralProperties as structProp

def controlStep(y, waypoint, mode):
    """
    arguments:
        arm : current arm object
        waypoint: desired state
        mode: fsm mode
    outputs:
        u   : control vector
    """

    # damping controllers

    # TODO: include other controllers
    # TODO: build total u vector

    if mode == 'none':
        u =  [0,0]
    elif mode == 'damping':
        y_rot = y['rot']

        r = y['r']
        I = structProp.getBoomInertia(r)

        ddtheta = torsionDampingControl(y_rot)
        M = I*ddtheta

        u = M   # assume we are commanding the moment

    return u


def torsionDampingControl(y):
    """
    Torsional Damping Controller

    Inputs:
        y: measurements [theta_start, theta_end, w_start, w_end]
    Outputs:
        u: ddtheta (d^2(theta)/dt^2
    """
    g = 1e-3

    w1 = y[2]
    w2 = y[3]

    u1 = 0
    u2 = -g*(w2-w1)
    u = np.array([u1,u2])
    return u
