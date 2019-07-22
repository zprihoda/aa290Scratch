import numpy as np

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
        u_torsion_damping = torsionDampingControl(y_rot)
        u = u_torsion_damping

    return u


def torsionDampingControl(y):
    g = 1e-3

    w1 = y[2]
    w2 = y[3]

    u1 = -g*y[2]
    u2 = 0
    u = np.array([u1,u2])
    return u
