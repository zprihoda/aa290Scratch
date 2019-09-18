import numpy as np
import scipy.linalg as spl

class Dynamics():
    def __init__(self, A, B, C=None):
        if C is None:
            C = np.eye(A.shape[0])

        self.A = A
        self.B = B
        self.C = C

    def discretizeDynamics(self,dt):

        A = self.A
        B = self.B

        t_arr = np.linspace(0,1,10)*dt
        y_arr = np.array([spl.expm(t*A) for t in t_arr])
        tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])

        A_d = spl.expm(A*dt)
        B_d = np.dot(tmp,B)

        return DiscreteDynamics(A_d,B_d,self.C)

class DiscreteDynamics():
    def __init__(self,A,B,C=None):
        if C is None:
            C = np.eye(A.shape[0])

        self.A = A
        self.B = B
        self.C = C
