

def simulateMeasurements(arm):
    """
    Simulate measurements and store them in a dictionary format

    key: desc
    rot: rotation of start and end of boom.  [theta_start, theta_end]
    """
    y = {}

    y['rot_z'] = arm.state.rot_z
    y['rate_z'] = arm.state.rate_z

    y['r'] = arm.r

    return y
