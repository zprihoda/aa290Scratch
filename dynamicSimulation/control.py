import numpy as np

def controlStep(arm, waypoint):
    """
    arguments:
        arm : current arm object
        waypoint: desired state
    outputs:
        u   : control vector
    """


    # damping controllers
    y_rot = [arm.state.rot_z[0],arm.state.rot_z[-1], arm.state.rate_z[0], arm.state.rate_z[-1]]       # we know the rotation angle and rate at the start and end of the boom
    u_torsion_damping = torsionDampingControl(y_rot)
    # TODO: include other controllers
    # TODO: build total u vector

    # u =  [0,0]
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
