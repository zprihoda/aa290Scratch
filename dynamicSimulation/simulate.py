import copy
import numpy as np

from control import controlStep
from dynamics import dynamicsStep
from measurement import simulateMeasurements
import structuralProperties as structProp
from plotResults import plotAll, animateTorsion, animateBending

def simulate(arm, traj, tf, dt_dyn, dt_control=None, u_inj=None):
    """
    Inputs:
        arm: arm object
        traj: desired trajectory
        tf: desired final simulation time
        dt: desired dt for control
        u_inj: control injection
    """

    if dt_control is None:
        dt_control = dt_dyn
    else:
        dt_control = max(np.floor(dt_control/dt_dyn)*dt_dyn, dt_dyn)
    T = int(dt_control/dt_dyn)

    state_list = []
    control_list = []

    t_steps = int(np.floor(tf/dt_dyn))
    t_arr = np.linspace(0,tf,t_steps+1)
    for i,t in enumerate(t_arr):
        print('Progress: {:.2f}%'.format(float(i)/(len(t_arr)-1) * 100), end='\r')

        # mode = finiteStateMachine(y,wp)
        # mode = 'none'
        # mode = 'damping'
        mode = 'mpc'

        if i%T == 0:
            j = i//T
            if j not in u_inj:
                wp = traj[:,j]
                y = simulateMeasurements(arm)
                u = controlStep(y,wp,mode,dt_control)
            else:
                u = {}
                u['rot'] = u_inj[j]['rot']
                u['lat'] = u_inj[j]['lat']

        # arm = dynamicsStep(arm,u,dt_dyn, noise_torsion=[1e-3,0], noise_bending=[1e-3,0,0,0])
        arm = dynamicsStep(arm,u,dt_dyn)

        state_list.append(copy.copy(arm.state))
        control_list.append(u)

    return state_list, control_list, t_arr



if __name__ == "__main__":
    from robotArm import Arm

    arm = Arm(r=1, theta_arr=np.zeros(6), num_fe=10)

    # simulate
    tf = 10.0

    dt_dyn = 1e-2
    dt_control = 1e-1
    t_steps = int(np.floor(tf/dt_control))
    t_arr = np.linspace(0,tf,t_steps+1)

    traj = np.zeros([2,len(t_arr)])

    u_inj = {0 : {'rot' : [1e-4,0],
                  'lat' : [1e-4,0]
                 }
            }

    state_list, control_list, t_arr = simulate(arm, traj, tf, dt_dyn, dt_control=dt_control, u_inj=u_inj)

    plotAll(state_list, control_list, t_arr)
    animateTorsion(state_list,t_arr)
    animateBending(state_list,t_arr)
