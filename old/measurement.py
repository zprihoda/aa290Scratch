

def simulateMeasurements(arm):
    """
    Simulate measurements and store them in a dictionary format

    key: desc
    rot: rotation of start and end of boom.  [theta_start, theta_end]
    """
    y = {}

    y['rot_z'] = arm.state.rot_z
    y['rate_z'] = arm.state.rate_z

    y['def_lat'] = arm.state.def_lat
    y['rate_lat'] = arm.state.rate_lat

    y['r'] = arm.r

    return y
