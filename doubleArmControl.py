import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from RobotArm import RobotArm2D


def animateDoubleArm(arm1,arm2,pos_arr1,pos_arr2,skip=1):
    fig, ax = plt.subplots()
    ax.set(xlim=[-10,10],ylim=[-10,10])

    base1 = arm1.base
    base2 = arm2.base

    # initialize lines
    arm1_line = ax.plot([base1[0],pos_arr1[0,0]], [base1[1],pos_arr1[1,0]],'b-')[0]
    arm2_line = ax.plot([base2[0],pos_arr2[0,0]], [base2[1],pos_arr2[1,0]],'r-')[0]

    def animate(i):
        arm1_line.set_xdata([base1[0],pos_arr1[0,skip*i]])
        arm1_line.set_ydata([base1[1],pos_arr1[1,skip*i]])
        arm2_line.set_xdata([base2[0],pos_arr2[0,skip*i]])
        arm2_line.set_ydata([base2[1],pos_arr2[1,skip*i]])

    anim = ani.FuncAnimation(fig, animate, interval=1.0, frames=np.arange(pos_arr1.shape[1]/skip))
    plt.draw()
    plt.show()


def main():
    arm1 = RobotArm2D(10,0, base=[-10,0])
    arm2 = RobotArm2D(10,np.pi, base=[10,0])

    dt = 0.01
    tf = 100
    t_arr = np.arange(0,tf+dt,dt)

    pos1 = np.zeros([2,len(t_arr)])
    pos2 = np.zeros([2,len(t_arr)])

    pos1[:,0] = arm1.getEndPos()
    pos2[:,0] = arm2.getEndPos()

    for i,t in enumerate(t_arr[0:-1]):

        r_des = np.array([np.sin(4*np.pi*t/tf)/2, np.sin(2*np.pi*t/tf)])
        v_des = 0.5*np.array([np.cos(4*np.pi*t/tf), np.cos(2*np.pi*t/tf)])

        arm1.controlVel(v_des,dt)
        arm2.controlVel(v_des,dt)

        pos1[:,i+1] = arm1.getEndPos()
        pos2[:,i+1] = arm2.getEndPos()

    # Plot Error
    err = npl.norm(pos1-pos2,axis=0)
    plt.figure()
    plt.plot(err)
    plt.xlabel('t')
    plt.ylabel('err (m)')
    plt.title('Relative Position Error of Arms')
    plt.show()

    animateDoubleArm(arm1, arm2, pos1, pos2, skip=max(1,int(0.1/dt)))


if __name__ == "__main__":
    main()
