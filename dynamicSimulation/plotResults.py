import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def plotTorsion(state_list, control_list, t_arr):
    rot_z_arr = np.array([state.rot_z for state in state_list])
    rate_z_arr = np.array([state.rate_z for state in state_list])
    control_arr = np.array([ctrl['rot'] for ctrl in control_list])

    n = state_list[0].n

    fig,axes = plt.subplots(3,1,sharex=True)
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
    plt.tight_layout()


def plotBending(state_list, control_list, t_arr):
    def_arr = np.array([state.def_lat for state in state_list])
    rate_arr = np.array([state.rate_lat for state in state_list])
    control_arr = np.array([ctrl['lat'] for ctrl in control_list])

    n = state_list[0].n

    fig,axes = plt.subplots(3,1,sharex=True)
    axes[0].plot(t_arr, def_arr[:,0])
    axes[0].plot(t_arr, def_arr[:,-2])
    axes[0].legend(['start','end'])
    axes[0].set_title('Bending Finite-Element Model')
    axes[0].set_ylabel(r'$\delta$')
    axes[0].grid()

    axes[1].plot(t_arr, rate_arr[:,0] - rate_arr[:,-2])
    axes[1].set_ylabel(r'$\dot{\delta}_1 - \dot{\delta}_2$')
    axes[1].grid()

    axes[2].plot(t_arr,control_arr)
    axes[2].set_ylabel('u')
    axes[2].set_xlabel('t')
    axes[2].grid()
    plt.tight_layout()

def animateTorsion(state_list, t_arr):

    dt = (t_arr[1]-t_arr[0])*1000   # in milliseconds

    fig, ax = plt.subplots()

    x = state_list[0].pos_z
    y = state_list[0].rot_z
    line, = ax.plot(x, y,'ro')

    ax.set_xlim(0, state_list[0].pos_z[-1])
    ax.set_ylim(np.min([state.rot_z for state in state_list]),
                np.max([state.rot_z for state in state_list]))
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\theta$')
    ax.grid()


    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        return line,

    def animate(i):
        line.set_ydata(state_list[i].rot_z)  # update the data.
        return line,

    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=range(len(state_list)), blit=True,
                        save_count=50, interval=dt)
    plt.show()


def animateBending(state_list, t_arr):

    dt = (t_arr[1]-t_arr[0])*1000   # in milliseconds

    fig, ax = plt.subplots()

    x = state_list[0].pos_z
    y = state_list[0].def_lat[0::2]
    line, = ax.plot(x, y,'ro')

    ax.set_xlim(0, state_list[0].pos_z[-1])
    ax.set_ylim(np.min([state.def_lat[0::2] for state in state_list]),
                np.max([state.def_lat[0::2] for state in state_list]))
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$u$')
    ax.grid()


    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        return line,

    def animate(i):
        w = state_list[i].def_lat[0::2]
        line.set_ydata(w)  # update the data.
        return line,

    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=range(len(state_list)), blit=True,
                        save_count=50, interval=dt*10)
    plt.show()


def plotAll(state_list, control_list, t_arr):
    plotTorsion(state_list, control_list, t_arr)
    plotBending(state_list, control_list, t_arr)
    plt.show()

