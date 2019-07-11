import numpy as np


## GLOBAL VARIABLES
AXIAL_STIFFNESS = 8000*175.127  # N/m
LATERAL_STIFFNESS = 2*175.127

AXIAL_STRENGTH = 90*4.44822     # N
LATERAL_STRENGTH = 0.5*4.44822

AXIAL_PRECISION = 0.002*0.0254  # m
LATERAL_PRECISION = 0.05*0.0254

SRPING_STIFFNESS = 100. # N/m

SPACECRAFT_MASS = 20.    # kg


def getStrain(F, k, L=1.0):
    """Returns the strain of the segment under a given force"""
    return (F/k)/L


def main():
    pass


if __name__ == "__main__":
    main()
