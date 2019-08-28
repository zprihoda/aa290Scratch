"""
See https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1232628
Approaches for dynamic modelling of flexible manipulator systems
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt

# np.set_printoptions(formatter={'float': lambda x: "{0:5.0f}".format(x)})

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


class LateralFEModel():
    def __init__(self, n, L, C_ratio=0, bc_start=2, bc_end=0):

        self.n = n
        self.L = L
        self.l = float(L)/n
        self.rho = 2710
        self.A = 0.0032*0.019
        self.E = 71e9
        self.I = 5.253e-11

        self.bc_start = bc_start
        self.bc_end = bc_end

        # obtain Finite Element model
        self.M_tot = np.zeros([2*n+1,2*n+1])
        self.K_tot = np.zeros([2*n+1,2*n+1])
        self.compileSystemMatrices()

        self.C_tot = C_ratio * (self.K_tot + self.M_tot)

        self.A, self.B = self.compileAB()

    @classmethod
    def getDynamics(cls, n, L, C_ratio=0, bc_start=2, bc_end=0):
        mdl = LateralFEModel(n, L, C_ratio=0, bc_start=2, bc_end=0)
        dyn = Dynamics(mdl.A, mdl.B)
        return dyn

    def compileSystemMatrices(self):

        # build total matrices out of element matrices
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

        # Apply boundary conditions
        idx = list(range(2*self.n+1))
        idx_rm = []
        if self.bc_start == 1:
            idx_rm.extend([1])
        if self.bc_start == 2:
            idx_rm.extend([1,2])

        if self.bc_end == 1:
            idx_rm.extend([-2])
        if self.bc_end == 2:
            idx_rm.extend([-2,-1])

        idx = np.array(list(set(idx)-set(idx_rm)))
        self.M_tot = self.M_tot[idx[:,None], idx]
        self.K_tot = self.K_tot[idx[:,None], idx]


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


class ReducedDynamics():
    def __init__(self, full_dyn, n_red):
        Af, Bf = full_dyn.A, full_dyn.B
        A_red, B_red, C_red = self.reduceDynamics(Af, Bf, n_red)

        self.A = A_red
        self.B = B_red
        self.C = C_red

    def reduceDynamics(self, Af, Bf, n_red, Cf=None):
        """
        See pages 78 and 209-211 of Approximation of Large-Scale Dynamical Systems by Antoulas
        """
        # get P and Q (infinite grammians)
        if Cf is None:
            Cf = np.eye(Af.shape[0])

        P = spl.solve_lyapunov(Af, -Bf @ Bf.conj().T)
        Q = spl.solve_lyapunov(Af.conj().T, -Cf.conj().T @ Cf)

        # Obtain U, K, Sigma
        U = npl.cholesky(P)
        lmbda,K = npl.eig(U.conj().T @ Q @ U)
        Sigma = np.diag(np.sqrt(lmbda))

        # Obtain balanced similarity transform matrices
        T = np.sqrt(Sigma) @ K.conj().T @ npl.inv(U)
        T_inv = U @ K @ (1/np.sqrt(Sigma))

        # Obtain Balanced Matrices
        A_bal = T @ Af @ T_inv
        B_bal = T @ B
        C_bal = C @ T_inv

        # Obtain reduced Matrices
        A_red = A_bal[:n_red,:n_red]
        B_red = B_bal[:n_red,:]
        C_red = C_bal[:,:n_red]

        return A_red, B_red, C_red


def main():
    dyn = LateralFEModel.getDynamics(n=5,L=0.9,C_ratio=1e-2)
    # dyn_red = ReducedDynamics(dyn,4)

    tf = 2.0
    dt = 0.001
    t_arr = np.arange(0,tf,dt)

    dyn_d = dyn.discretizeDynamics(dt)
    A_d, B_d = dyn_d.A, dyn_d.B

    # print(npl.eig(dyn.A)[0])
    # input()

    X = np.zeros(A_d.shape[0])

    u_arr = np.zeros([2,len(t_arr)])
    X_arr = np.zeros([len(X),len(t_arr)])
    X_arr[:,0] = X
    u_arr[0,200:500] = 0.1
    u_arr[0,500:800] = -0.1

    for i in range(len(t_arr)-1):
        u = u_arr[:,i]
        X = A_d@X + B_d@u
        X_arr[:,i+1] = X

    plt.plot(t_arr,X_arr[0,:])
    plt.show()



if __name__ == "__main__":
    main()
