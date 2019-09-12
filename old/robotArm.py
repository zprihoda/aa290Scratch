import numpy as np

"""
Robotic Arm Module

References:
1: http://www.its.caltech.edu/~sslab/PUBLICATIONS/LECLERC_Ultra_Thin_Composite_Deployable_Booms_Full.pdf
"""

# density_GSM = 17.    # grams per square meter [1]
density_GSM = 1700.    # grams per square meter (debugging value)
r = 1.5/100
t = 1e-3

class Arm():
    ##Class Variables
    structProps = {}
    structProps['k_rot'] = 40./1000    # N*m/rad
    structProps['c_rot'] = 0.0005*structProps['k_rot']

    structProps['k_lat'] = 350.    # N/m
    structProps['c_lat'] = 0.0005*structProps['k_lat']

    structProps['density'] = density_GSM/1000 / t  # kg/m^3
    structProps['radius'] = r
    structProps['thickness'] = t


    def __init__(self, r, theta_arr, num_fe=None):
        self.r = float(r)
        self.theta_arr = theta_arr

        if num_fe is None:
            num_fe = int(round(r*10))
        self.state = ArmState(num_fe,r)


class ArmState():
    def __init__(self,n,r):
        """
        Describe the state of the arm

        instance variables
        pos_z = [z_1, ... , z_n] : position of center of each finite element (from start of arm)
        rot_z = [theta_1, ... , theta_n, theta_dot_1, ... , theta_dot_n] : rotational state of each finite element
        rate_z = [theta_dot_1 ...] : rotation rates
        def_x = [x_1, ... , x_n] : x deflection of each FE
        def_y = [y_1, ... , y_n] : y deflection of each FE
        """

        dl = float(r)/n

        self.n = n
        self.dl = dl

        self.pos_z = np.arange(dl/2.,r,dl)

        self.rot_z = np.zeros(n)    # [theta_1 ... theta_n]
        self.rate_z = np.zeros(n)   # [dtheta_1, ... dtheta_n]

        self.def_lat = np.zeros(2*n)    # [w_1, theta_1, ... , w_n, theta_n]
        self.rate_lat = np.zeros(2*n)


if __name__ == "__main__":
    arm = Arm(1,np.zeros(6))
    print(arm.state.pos_z)
