
def controlStep(arm, waypoint):
    """
    arguments:
        arm : current arm object
        waypoint: desired state
    outputs:
        u   : control vector
    """

    # damping controllers
    # y_rot = [arm.state.rotz[0],arm.state.rotz[-1], arm.state.ratez[0], arm.state.ratez[-1]]       # we know the rotation angle and rate at the start and end of the boom
    # u_torsion_damping = torsionDampingControl(y_rot)
    # TODO: include other controllers
    # TODO: build total u vector

    u =  [0,0]
    return u

