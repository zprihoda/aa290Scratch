import numpy as np

class Arm():
    def __init__(self, r, theta_arr, num_fe=None):
        self.r = r
        self.theta_arr = theta_arr

        if num_fe is None:
            num_fe = int(round(r*10))
        self.state = ArmState(num_fe,r)

        self.structProps = {}
        self.structProps['k_rot'] = 1
        self.structProps['c_rot'] = 0.001

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

        self.rot_z = np.zeros(n)
        self.rate_z = np.zeros(n)

        self.def_x = np.zeros(n)
        self.def_y = np.zeros(n)


if __name__ == "__main__":
    arm = Arm(1,np.zeros(6))
    print arm.state.pos_z
