import numpy as np
import plottingTools as pt

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
    t_f = 10.0
    dt_dyn = 0.01
    dt_control = 0.1

    n_fe = 100
    n_full = 4*n_fe - 4 + 2  # v,theta + d(v,theta) - bc + hub
    n_red = 10
    x0 = np.zeros(n_full)
    x0[-1] = 0.1
    x_des = np.zeros(n_full)
    x_des[0] = np.pi/2      # 90 degree rotation, zero bending

    # setup matrices
    t_arr = np.arange(0,t_f,dt_dyn)
    x_arr = np.zeros([n_full,len(t_arr)+1])
    u_arr = np.zeros([2,len(t_arr)])
    x_arr[:,0] = x0

    # setup dynamics (and discretize them for the simulation)
    print("Compiling Dynamic Matrices...")
    dyn_f = LateralFEModel.getDynamics(n=n_fe, L=0.9, C_ratio=1e-4)
    dyn_df = dyn_f.discretizeDynamics(dt_dyn)
    dyn_r = reduceDynamics(dyn_f, n_red=n_red)
    dyn_dr = dyn_r.discretizeDynamics(dt_control)

    # simulate
    print("Running Simulation...")
    x = x0
    for i, t in enumerate(t_arr):
        # y = getMeasurement(dyn_f, x)

        # TODO: Implement controller
        u = np.zeros(2)

        x = dynStep(dyn_df, x, u)

        # store variables
        x_arr[:,i+1] = x
        u_arr[:,i] = u


    # plot results
    pt.plotHubAngle(x_arr, u_arr, t_arr)


if __name__ == "__main__":
    main()
