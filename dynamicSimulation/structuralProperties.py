import numpy as np

BOOM_THICKNESS = 1e-3   # m
BOOM_RADIUS = 1.5/100   # m
BOOM_DENSITY = 1500     # kg/m^3

def getBoomInertia(l):
    """
    Determine moments of inertia of Boom at a specified length about one end

    args:
        l (float) :  length of boom
    outputs:
        inertia matrix (3x3) for x,y (transverse) and z (axial) directions
    """

    r = BOOM_RADIUS
    delta = BOOM_THICKNESS
    rho = BOOM_DENSITY

    m = 2*np.pi*r*delta*l * rho
    I_xx = m/6 * (3*r**2 + 2*l**2)
    I_yy = I_xx
    I_zz = m*r**2

    return np.diag([I_xx, I_yy, I_zz])


def getTotalInertia(l_boom, I_objects, m_objects):
    """
    Determine Total Moment of Inertia for a single boom including external objects
    Could include end effector or an object we are manipulating

    args:
        l_boom (float) : length of boom
        I_objects (list of 3x3 arrays) : moments of inertia for each object
        m_objects (list of 3x1 arrays) : mass of each object
    outputs:
        inertia matrix (3x3) for x,y (transverse) and z (axial) directions
    """

    I_boom = getBoomInertia(l_boom)

    I_tot = I_boom
    for I_cm,m in zip(I_objects, m_objects):
        I = I_cm + m*np.diag([l_boom,l_boom,0])**2

    return I_tot
