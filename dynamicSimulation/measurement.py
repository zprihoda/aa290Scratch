

def simulateMeasurements(arm):
    """
    Simulate measurements and store them in a dictionary format

    key: desc
    rot: rotation of start and end of boom.  [theta_start, theta_end]
    """
    y = {}

    y['rot'] = [arm.state.rot_z[0],arm.state.rot_z[-1],
                arm.state.rate_z[0], arm.state.rate_z[-1]]     # we know the rotation angle and rate at the start and end of the boom
    y['r'] = arm.r

    return y
