"""
See https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1232628
Approaches for dynamic modelling of flexiblemanipulator systems
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

# np.set_printoptions(formatter={'float': lambda x: "{0:5.0f}".format(x)})

class Dynamics():
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C


class TorsionFEModel():
    def __init__(self, n, L, C_ratio=0.01):

        self.n = n
        self.L = L
        self.l = float(L)/n
        self.rho = 1
        self.A = 0.1
        self.E = 1
        self.I = 1

        self.M_tot = np.zeros([2*n+1,2*n+1])
        self.K_tot = np.zeros([2*n+1,2*n+1])
        self.compileSystemMatrices()

        idx = np.array([0]+list(range(3,2*n+1)))
        self.M_tot = self.M_tot[idx[:,None], idx]
        self.K_tot = self.K_tot[idx[:,None], idx]

        self.C_tot = C_ratio * (self.K_tot + self.M_tot)

        self.A, self.B = self.compileAB()

    def compileSystemMatrices(self):

        for k in np.arange(1,self.n):

            M_e = self.getElementMassMatrix(k)
            K_e = self.getElementStiffnessMatrix(k)

            start = 2*(k-1) + 1
            stop  = 2*(k+1) + 1

            self.M_tot[0,0] += M_e[0,0]
            self.M_tot[0,start:stop] += M_e[0,1:]
            self.M_tot[start:stop,0] += M_e[0,1:]
            self.M_tot[start:stop,start:stop] += M_e[1:,1:]

            self.K_tot[start:stop,start:stop] += K_e

    def getElementMassMatrix(self,k):
        # obtain constants
        l = self.l
        c = self.rho * self.A * self.l / 420

        # mass terms
        m11 = 140*l**2 * (3*k**2 - 3*k + 1)
        m12 = 21*l * (10*k - 7)
        m13 = 7*l**2 * (5*k - 3)
        m14 = 21*l * (10*k - 3)
        m15 = -7*l**2 * (5*k - 2)
        m = [m11,m12,m13,m14,m15]

        # form element matrix
        M_k = np.zeros([5,5])
        M_k[:,0] = m
        M_k[0,:] = m
        M_k[1:,1:] = np.array([
            [   156,    22*l,    54,   -13*l],
            [  22*l,  4*l**2,  13*l, -3*l**2],
            [    54,    13*l,   156,   -22*l],
            [ -13*l, -3*l**2, -22*l,  4*l**2]
            ])
        M_k *= c

        return M_k

    def getElementStiffnessMatrix(self,k):
        l = self.l
        c = self.E*self.I/l**3

        K_k = np.array([
            [  12,    6*l,  -12,    6*l],
            [ 6*l, 4*l**2, -6*l, 2*l**2],
            [ -12,   -6*l,   12,   -6*l],
            [ 6*l, 2*l**2, -6*l, 4*l**2]
            ])
        K_k *= c

        return K_k

    def compileAB(self):
        # load parameters
        M = self.M_tot
        K = self.K_tot
        C = self.C_tot

        M_inv = npl.inv(M)

        n = M.shape[0]

        # form continuous dynamics
        A = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]),
                       np.hstack([-M_inv@K, -M_inv@C])])
        B_tmp = np.vstack([np.array([1,0]),
                           np.zeros([n-2,2]),
                           np.array([0,1])
                          ])
        B = np.vstack([np.zeros([n,2]),
                       M_inv@B_tmp])

        return A,B

def discretizeAB(A,B,dt):

    t_arr = np.linspace(0,1,10)*dt
    y_arr = np.array([spl.expm(t*A) for t in t_arr])
    tmp = np.sum(y_arr,axis=0)*(t_arr[1]-t_arr[0])

    A_d = spl.expm(A*dt)
    B_d = np.dot(tmp,B)

    return A_d, B_d

def main():
    mdl = TorsionFEModel(n=5,L=1)

    tf = 0.1
    dt = 0.0001
    t_arr = np.arange(0,tf,dt)

    A_d, B_d = discretizeAB(mdl.A, mdl.B,dt)

    X = np.zeros(A_d.shape[0])

    u_arr = np.zeros([2,len(t_arr)])
    X_arr = np.zeros([len(X),len(t_arr)])
    X_arr[:,0] = X
    u_arr[0,0:10] = 0.1
    u_arr[0,10:20] = -0.1

    for i in range(len(t_arr)-1):
        u = u_arr[:,i]
        X = A_d@X + B_d@u
        X_arr[:,i+1] = X

    plt.plot(t_arr,X_arr[-2,:])
    plt.show()



if __name__ == "__main__":
    main()
