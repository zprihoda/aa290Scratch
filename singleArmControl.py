import numpy as np

from RobotArm import RobotArm1D
from plottingTools import animateArm1D


def main():
	arm = RobotArm1D(10,0)

	dt = 0.1
	tf = 100
	t_arr = np.arange(0,tf+dt,dt)

	r_arr = np.zeros(len(t_arr))
	theta_arr = np.zeros(len(t_arr))

	r_arr[0] = arm.r
	theta_arr[0] = arm.theta
	for i,t in enumerate(t_arr[0:-1]):
		v_des = [np.sin(2*np.pi*t/tf), np.cos(2*np.pi*t/tf)]

		arm.control(v_des,dt)

		r_arr[i+1] = arm.r
		theta_arr[i+1] = arm.theta

	animateArm1D(r_arr,theta_arr)


if __name__ == "__main__":
	main()
