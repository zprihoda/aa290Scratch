import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np


def plotResults(X_arr, u_arr, t_arr):

    # obtain x,y,theta coordinates from state
    n = (len(X_arr[:,0])-2)//4
    v_arr = X_arr[1:2*n:2,:]   # deflection at each element at each time
    theta_arr = X_arr[0,:]  # hub angle at each time
    u = np.linspace(0, 2, n+1)

    x_arr = np.zeros([len(t_arr)])
    y_arr = np.zeros([len(t_arr)])
    for i in range(len(t_arr)):
        v = np.hstack([0,v_arr[:,i]])
        theta = theta_arr[i]

        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        tmp = np.vstack([u,v])
        tmp2 = np.dot(R,tmp)
        x = tmp2[0,:]
        y = tmp2[1,:]

        x_arr[i] = x[-1]
        y_arr[i] = y[-1]

    # plot x,y vs time
    plt.figure()
    plt.plot(t_arr,x_arr,'b-')
    plt.plot(t_arr,y_arr,'o-')
    plt.plot([t_arr[0],t_arr[-1]], [0,0],'b--')
    plt.plot([t_arr[0],t_arr[-1]], [2,2],'o--')
    plt.title('Payload Position vs Time')
    plt.xlabel('t (s)')
    plt.ylabel('position (m)')
    plt.legend(['x','y'])
    plt.grid()

    plt.savefig('plots/position.png')
    plt.close()

    # plot deflection of end point vs time
    n = len(X_arr)
    plt.figure()
    plt.title('Endpoint Deflection vs Time')
    plt.plot(t_arr,X_arr[n//2-1,:-1])
    plt.xlabel('t (s)')
    plt.ylabel('v (m)')
    plt.grid()

    plt.savefig('plots/EndPointDef.png')
    plt.close()

    # plot control vs time
    plt.figure()
    plt.title('Control Input vs Time')
    plt.plot(t_arr,u_arr[0,:])
    plt.plot(t_arr,u_arr[1,:])
    plt.legend(['Start','end'])
    plt.xlabel('t (s)')
    plt.ylabel('u (Nm)')
    plt.grid()

    plt.savefig('plots/Control.png')
    plt.close()

    # plt.show()

def animateResults(X_arr, t_arr):

    # obtain x,y,theta coordinates from state
    n = (len(X_arr[:,0])-2)//4
    v_arr = X_arr[1:2*n:2,:]   # deflection at each element at each time
    theta_arr = X_arr[0,:]  # hub angle at each time
    u = np.linspace(0, 2, n+1)

    x_arr = np.zeros([len(u),len(t_arr)])
    y_arr = np.zeros([len(u),len(t_arr)])
    for i in range(len(t_arr)):
        v = np.hstack([0,v_arr[:,i]])
        theta = theta_arr[i]

        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        tmp = np.vstack([u,v])
        tmp2 = np.dot(R,tmp)
        x = tmp2[0,:]
        y = tmp2[1,:]

        x_arr[:,i] = x
        y_arr[:,i] = y

    # plot stuff
    fig, ax = plt.subplots()

    x_max = np.max(x_arr)
    x_min = np.min(x_arr)
    y_max = np.max(y_arr)
    y_min = np.min(y_arr)

    coord_max = max(x_max,y_max)
    coord_min = min(x_min,y_min)

    c = (coord_max + coord_min) / 2
    d = 1.1 * (coord_max - coord_min) / 2

    # initialize plot
    line = ax.plot([],[],'b.-')[0]
    ax.set(xlim=[c-d, c+d],ylim=[c-d, c+d])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title("Setpoint Maneuver")

    def animate(i):
        x = x_arr[:,i]
        y = y_arr[:,i]
        line.set_xdata(x)
        line.set_ydata(y)

    # generate animation
    anim = ani.FuncAnimation(fig, animate, frames=np.arange(0,len(t_arr),2), interval=20)
    anim.save("plots/full.mp4")
    plt.close()

def animateBending(X_arr, t_arr):

    # obtain deflection from state
    n = (len(X_arr[:,0])-2)//4
    v_arr = X_arr[1:2*n:2,:]   # deflection at each element at each time
    theta_arr = X_arr[0,:]  # hub angle at each time
    u = np.linspace(0, 2, n+1)

    y_arr = np.zeros([len(u),len(t_arr)])
    for i in range(len(t_arr)):
        v = np.hstack([0,v_arr[:,i]])
        y_arr[:,i] = -v

    # plot stuff
    fig, ax = plt.subplots()

    y_max = np.max(y_arr)
    y_min = np.min(y_arr)

    c = (y_max + y_min) / 2
    d = 1.1 * (y_max - y_min) / 2

    # initialize plot
    line = ax.plot([],[],'b.-')[0]
    ax.set(xlim=[0, max(u)],ylim=[c-d, c+d])
    ax.set_xlabel('u (m)')
    ax.set_ylabel('v (m)')
    ax.set_title("Arm Deflection")

    def animate(i):
        x = u
        y = y_arr[:,i]
        line.set_xdata(x)
        line.set_ydata(y)

    # generate animation
    anim = ani.FuncAnimation(fig, animate, frames=np.arange(0,len(t_arr),2), interval=20)
    anim.save("plots/bending.mp4")
    plt.close()
