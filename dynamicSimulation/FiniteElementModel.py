"""
See https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1232628
Approaches for dynamic modelling of flexible manipulator systems

stabsep
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl

from reducedDynamics import *
from dynamics import Dynamics

class LateralFEModel():
    def __init__(self, n, L, C_ratio=0, bc_start=2, bc_end=0, m_obj=None, I_obj=None):
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
        if m_obj is not None and I_obj is not None:
            M_obj = np.diag(np.hstack([I_obj+m_obj*L**2, np.zeros(2*n-2), m_obj, I_obj]))
            self.M_tot += M_obj
        self.applyBoundaryConditions()
        self.C_tot = C_ratio * self.K_tot


        self.A, self.B = self.compileAB()

    @classmethod
    def getDynamics(cls, n, L, C_ratio=0, bc_start=2, bc_end=0, m_obj=None, I_obj=None):
        mdl = LateralFEModel(n, L, C_ratio=C_ratio, bc_start=bc_start, bc_end=bc_end, m_obj=m_obj, I_obj=I_obj)
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

    def applyBoundaryConditions(self):
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


def main():
    dyn = LateralFEModel.getDynamics(n=100,L=0.9,C_ratio=1e-4)

    tf = 2.0
    dt = 0.001
    t_arr = np.arange(0,tf,dt)

    dyn_d = dyn.discretizeDynamics(dt)
    A_d, B_d = dyn_d.A, dyn_d.B

    X = np.zeros(A_d.shape[0])

    u_arr = np.zeros([2,len(t_arr)])
    X_arr = np.zeros([len(X),len(t_arr)])
    X_arr[:,0] = X
    u_arr[0,200:500] = 0.2
    u_arr[0,500:800] = -0.2

    for i in range(len(t_arr)-1):
        u = u_arr[:,i]
        X = A_d@X + B_d@u
        X_arr[:,i+1] = X

    from plottingTools import plotHubAngle, animateResults
    plotHubAngle(X_arr,u_arr,t_arr)
    animateResults(X_arr, t_arr)



if __name__ == "__main__":
    main()
