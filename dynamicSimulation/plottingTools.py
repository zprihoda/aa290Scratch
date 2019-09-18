import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

def plotHubAngle(X_arr, u_arr, t_arr):
    # plot hub angle vs time
    plt.plot(t_arr,np.rad2deg(X_arr[0,:-1]))
    plt.xlabel('t')
    plt.ylabel(r'$\theta$')
    plt.show()

def animateResults(X_arr, t_arr):

    # obtain x,y,theta coordinates from state
    n = (len(X_arr[:,0])-2)//4
    v_arr = X_arr[1:2*n:2,:]   # deflection at each element at each time
    theta_arr = X_arr[0,:]  # hub angle at each time
    u = np.linspace(0, 1, n+1)

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

    def animate(i):
        x = x_arr[:,i]
        y = y_arr[:,i]
        line.set_xdata(x)
        line.set_ydata(y)

    # generate animation
    anim = ani.FuncAnimation(fig, animate, frames=np.arange(0,len(t_arr),2), interval=20)
    plt.show()

