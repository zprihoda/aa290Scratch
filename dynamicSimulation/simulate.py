import copy
import numpy as np
import pylab as pl

from control import controlStep
from dynamics import dynamicsStep
from measurement import simulateMeasurements
import structuralProperties as structProp

def simulate(arm, traj, tf, dt, u_inj=None):
    """
    Inputs:
        arm: arm object
        traj: desired trajectory
        tf: desired final simulation time
        dt: desired dt for control
        u_inj: control injection
    """

    state_list = []
    control_list = []

    t_steps = int(np.floor(tf/dt))
    t_arr = np.linspace(0,tf,t_steps+1)
    for i,t in enumerate(t_arr):
        print('Progress: {:.2f}%'.format(float(i)/(len(t_arr)-1) * 100), end = '\r')
        y = simulateMeasurements(arm)

        # mode = finiteStateMachine(y,wp)
        # mode = 'none'
        # mode = 'damping'
        mode = 'mpc'

        if i not in u_inj:
            wp = traj[:,i]
            u = controlStep(y,wp,mode,dt)
        else:
            u = u_inj[i]

        arm = dynamicsStep(arm,u,dt)

        state_list.append(copy.copy(arm.state))
        control_list.append(u)

    plotResults(state_list, control_list, t_arr)


def plotResults(state_list, control_list, t_arr):

    rot_z_arr = np.array([state.rot_z for state in state_list])
    rate_z_arr = np.array([state.rate_z for state in state_list])
    control_arr = np.array(control_list)

    n = state_list[0].n

    fig,axes = pl.subplots(3,1,sharex=True)
    axes[0].plot(t_arr, rot_z_arr[:,0])
    axes[0].plot(t_arr, rot_z_arr[:,-1])
    axes[0].legend(['start','end'])
    axes[0].set_title('Torsional Finite-Element Model')
    axes[0].set_ylabel(r'$\theta$')
    axes[0].grid()

    axes[1].plot(t_arr, rate_z_arr[:,0] - rate_z_arr[:,-1])
    axes[1].set_ylabel(r'$\dot{\theta}_1 - \dot{\theta}_2$')
    axes[1].grid()

    axes[2].plot(t_arr,control_arr)
    axes[2].set_ylabel('u')
    axes[2].set_xlabel('t')
    axes[2].grid()
    pl.tight_layout()

    pl.show()

if __name__ == "__main__":
    from robotArm import Arm

    arm = Arm(r=1, theta_arr=np.zeros(6), num_fe=10)

    # simulate
    tf = 2.0

    dt = 1e-2
    t_steps = int(np.floor(tf/dt))
    t_arr = np.linspace(0,tf,t_steps+1)

    traj = np.zeros([2,len(t_arr)])

    u_inj = {0: [1e-3,0]}

    simulate(arm, traj, tf, dt, u_inj=u_inj)
