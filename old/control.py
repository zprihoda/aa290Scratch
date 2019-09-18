
"""
1: https://en.wikibooks.org/wiki/Strength_of_Materials/Torsion
    Torsional Strength Equation:
        phi = TL/(Gj)

TODO: Implement Max transverse stress constraint
TODO: Implement variable boundary conditions to mpc controllers
TODO: Consider LQR for speed improvements
TODO: Consider running MPC at 1 HZ (or something)
        and keeping the 10 most recent controls
"""

import numpy as np
import structuralProperties as structProp
import dynamics

import cvxpy as cp

from robotArm import Arm

class Controller():
    def __init__(self, dyn):
        self.dyn = dyn

    def controlStep(self, y, waypoint, mode):
        """
        arguments:
            arm : current arm object
            waypoint: desired state
            mode: fsm mode
        outputs:
            u   : control dictionary
        """

        # TODO: include other controllers
        # TODO: build total u vector

        u = {}

        if mode == 'none':
            u['rot'] =  [0,0]
            u['lat'] = [0,0,0,0]

        elif mode == 'mpc':
            u['rot'] = self.mpcRotController(y)
            u['lat'] = self.mpcLatController(y)

        return u

    def mpcRotController(self, y):
        A = self.dyn.A_torsion
        B = self.dyn.B_torsion
        dt = self.dyn.dt

        k_rot = self.dyn.structProps['k_rot']

        n = len(y['rot_z'])
        dl = y['r']/n

        ## setup optimal control problem
        X0 = np.append(y['rot_z'], y['rate_z'])
        X_des = np.zeros(2*n)
        X_des[0:n] = np.pi
        T = 50  # time horizon

        # solve cvx problem
        z = cp.Variable([2*n,T])
        u = cp.Variable([2,T-1])

        g_c = 1     # control
        g_f = 1     # final

        u_max = 1e-3    # max control
        T_max = 100     # max torsion

        J_c = cp.sum(cp.abs(u))                     # control
        J_f = cp.norm(z[:,-1]-X_des)                # final
        obj = cp.Minimize(g_c*J_c + g_f*J_f)

        constr = []
        constr += [z[:,0] == X0]                    # initial condition
        constr += [z[:,1:] == A@z[:,:-1] + B@u]     # FE dynamics
        constr += [cp.abs(u) <= u_max]              # control limits
        constr += [cp.abs(z[0:n-1,:] - z[1:n,:])  <= T_max*dl/k_rot] #torsional strength [1]

        prob = cp.Problem(obj, constr)
        try:
            result = prob.solve()
            if prob.status == 'optimal':
                u_opt = u.value[:,0]
            else:
                u_opt = [0,0]
        except cp.error.SolverError:
            print("Warning: cvx solver failed")
            # TODO: should be able to extract the best control from
            #   cvxpy when it fails. OR, just use a solver that doesn't fail
            #   e.g. MOSEK
            u_opt = [0,0]

        return u_opt

    def mpcLatController(self,y):
        A = self.dyn.A_bending
        B = self.dyn.B_bending
        B = B[:,0::2]

        n = self.dyn.n

        # handle fixed deflection bc
        idx_rm = [0]
        idx_keep = np.array(list(set(range(2*n))-set(idx_rm)))
        idx_keep2 = np.append(idx_keep, idx_keep+2*n)
        idx_rm = np.array(idx_rm)
        idx_rm2 = np.append(idx_rm, idx_rm+2*n)

        A = A[idx_keep2[:,None],idx_keep2]
        B = B[idx_keep2,:]

        ## setup optimal control problem
        X0 = np.append(y['def_lat'], y['rate_lat'])
        X_des = np.zeros(4*n)
        T = 50  # time horizon

        # solve cvx problem
        z = cp.Variable([4*n,T])
        u = cp.Variable([2,T-1])

        g_c = 1     # control
        g_f = 2     # final

        u_max = 1e-3      # max control
        T_max = 100       # max transverse stress

        J_c = cp.sum(cp.abs(u))                     # control
        J_f = cp.norm(z[:,-1]-X_des)                # final
        obj = cp.Minimize(g_c*J_c + g_f*J_f)

        constr = []
        constr += [z[:,0] == X0]                    # initial condition
        constr += [z[idx_keep2,1:] == A@z[idx_keep2,:-1] + B@u]     # FE dynamics
        constr += [z[idx_rm2,:] == 0]
        constr += [cp.abs(u) <= u_max]              # control limits

        prob = cp.Problem(obj, constr)
        try:
            result = prob.solve(feastol=1e-3, abstol=1e-3, reltol=5e-1)
            if prob.status == 'optimal':
                u_opt = u.value[:,0]
            else:
                u_opt = [0,0]
        except cp.error.SolverError:
            print("Warning: cvx solver failed")
            # TODO: should be able to extract the best control from
            #   cvxpy when it fails. OR, just use a solver that doesn't fail
            #   e.g. MOSEK
            u_opt = [0,0]

        return u_opt
