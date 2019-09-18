import numpy as np


def getBoomInertia(l,r,delta,rho):
    """
    Determine moments of inertia of Boom at a specified length about one end

    args:
        l (float)       :   length of boom
        r (float)       :   radius of boom
        delta (float)   :   thickness of boom
        rho (float)     :   density of boom
    outputs:
        inertia matrix (3x3) for x,y (transverse) and z (axial) directions
    """

    m = 2*np.pi*r*delta*l * rho
    I_xx = m/6 * (3*r**2 + 2*l**2)
    I_yy = I_xx
    I_zz = m*r**2

    return np.diag([I_xx, I_yy, I_zz])

