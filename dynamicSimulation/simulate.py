import numpy as np
import plottingTools as pt

from control import Controller
from FiniteElementModel import LateralFEModel
from reducedDynamics import reduceDynamics

def dynStep(dyn, x, u):
    x_new = dyn.A @ x + dyn.B @ u
    return x_new

def getMeasurement(dyn, x):
    y = dyn.C @ x
    return y

def main():
    # simulation parameters
    t_f = 30.0
    dt_dyn = 0.01
    dt_control = 0.1

    n_fe = 100
    n_red = 50

    n_full = 4*n_fe - 4 + 2  # v,theta + d(v,theta) - bc + hub (do not change)

    x0 = np.zeros(n_full)
    x0[-1] = 0.1
    x_des = np.zeros(n_full)
    x_des[0] = np.pi/2      # 90 degree rotation, zero bending

    T = 10  # time horizon for controller (in steps)

    L = 2.0
    m_obj = 1  # kg
    I_obj = 1/6 * m_obj * 0.5**2    # cube with sides of length 0.5 m

    # setup matrices
    t_arr = np.arange(0,t_f,dt_dyn)
    x_arr = np.zeros([n_full,len(t_arr)+1])
    u_arr = np.zeros([2,len(t_arr)])
    x_arr[:,0] = x0

    # setup dynamics (and discretize them for the simulation)
    print("Compiling Dynamic Matrices...")
    # dyn_f = LateralFEModel.getDynamics(n=n_fe, L=L, C_ratio=1e-4)
    dyn_f = LateralFEModel.getDynamics(n=n_fe, L=L, C_ratio=1e-4, m_obj=m_obj, I_obj=I_obj)
    dyn_df = dyn_f.discretizeDynamics(dt_dyn)
    dyn_r = reduceDynamics(dyn_f, n_red=n_red, debug=0)
    dyn_dr = dyn_r.discretizeDynamics(dt_control)

    # setup controller
    controller = Controller(dyn_dr, T, x_des)

    # simulate
    print("Running Simulation...")
    x = x0
    u_traj = None
    xr_traj = None

    control_cycle = int(np.ceil(dt_control/dt_dyn))

    for i, t in enumerate(t_arr):
        print('Progress: {:.2f}%'.format(float(i)/(len(t_arr)-1) * 100), end='\r')

        # y = getMeasurement(dyn_f, x)

        # determine control
        if i % control_cycle == 0:  # update control
            u_ctrl, xr_ctrl = controller.control(x)
            if u_ctrl is not None:
                t_ref = t
                u_traj = u_ctrl
                xr_traj = xr_ctrl

        if u_traj is not None:
            idx = int((t-t_ref) // dt_control)
            u = u_traj[:,idx]
        else:
            u = np.zeros(2)

        # simulate dynamics
        x = dynStep(dyn_df, x, u)

        # store variables
        x_arr[:,i+1] = x
        u_arr[:,i] = u


    # plot results
    pt.plotResults(x_arr, u_arr, t_arr)
    pt.animateResults(x_arr, t_arr)


if __name__ == "__main__":
    main()
