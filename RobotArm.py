import numpy as np
import numpy.linalg as npl


class MultiArmRobot2D():
    def __init__(self):
        self.arm_dict  = {}
        self.arm_fixed = {}

    def addArm(self,arm,arm_id,fixed=False):
        self.arm_dict[arm_id] = arm

        if fixed:
            self.arm_fixed[arm_id] = arm.getEndPos()
        else:
            self.arm_fixed[arm_id] = None

    def fixArm(self,arm_id):
        self.arm_fixed[arm_id] = self.arm_dict[arm_id].getEndPos()

    def releaseArm(self,arm_id):
        self.arm_fixed[arm_id] = None


class RobotArm2D():
    def __init__(self,r,theta,base=[0,0]):
        """
        2D Robotic Arm
        Arguments:
            r: initial length of arm
            theta: rotation angle (measured from counter-clockwise from x)
            base: specifies where the base of the arm is attached
        """
        self.r = r
        self.theta = theta
        self.base = np.array(base)

    def getEndPos(self):
        x = self.r*np.cos(self.theta) + self.base[0]
        y = self.r*np.sin(self.theta) + self.base[1]
        return np.array([x,y])

    def controlVel(self,v,dt):
        r = self.r
        theta = self.theta

        B = np.array([[np.cos(theta), -r*np.sin(theta)],
                      [np.sin(theta), r*np.cos(theta)]])

        u = npl.solve(B,v)
        r_dot = u[0]
        theta_dot = u[1]

        self.r = r + r_dot*dt
        self.theta = theta + theta_dot*dt
        return r_dot, theta_dot

    def controlRaw(self,r_dot,theta_dot,dt):
        self.r = self.r + r_dot*dt
        self.theta = self.theta + theta_dot*dt
