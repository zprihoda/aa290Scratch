import cvxpy as cp

class Controller():
    def __init__(self, dyn, T, x_des):
        self.dyn = dyn
        self.T = T  # time horizon

        ## setup optimization problem
        n = self.dyn.A.shape[0]
        m = self.dyn.B.shape[1]

        # setup variable
        self.z = cp.Variable([n,T])
        self.u = cp.Variable([m,T-1])

        # objective function
        g_c = 0     # control
        g_f = 1     # final

        xr_des = dyn.reduceState(x_des)

        J_c = cp.sum(cp.abs(self.u))                     # control
        J_f = cp.norm(self.z[:,-1]-xr_des)                # final
        self.obj = cp.Minimize(g_c*J_c + g_f*J_f)

        # constraints
        u_max = 1.0
        self.constr = []
        self.constr += [self.z[:,1:] == dyn.A @ self.z[:,:-1] + dyn.B @ self.u]     # FE dynamics
        self.constr += [cp.abs(self.u) <= u_max]                          # control limits

    def control(self, x0):
        xr0 = self.dyn.reduceState(x0)
        constr = [self.z[:,0] == xr0]                    # initial condition
        constr += self.constr
        prob = cp.Problem(self.obj, constr)

        z_opt = None
        u_opt = None

        try:
            result = prob.solve()
            if prob.status == 'optimal':
                z_opt = self.z.value
                u_opt = self.u.value
        except cp.error.SolverError:
            print("Warning: cvx solver failed")

        return u_opt, z_opt
