import numpy as np
import numpy.linalg as npl

from RobotArm import RobotArm2D


def main():
	arm1 = RobotArm2D(np.sqrt(2)*5,np.pi/4, base=[0,0])
	arm2 = RobotArm2D(np.sqrt(2)*5,3*np.pi/4, base=[10,0])

	dt = 0.01
	tf = 100
	t_arr = np.arange(0,tf+dt,dt)

	r1_arr = np.zeros(len(t_arr))
	theta1_arr = np.zeros(len(t_arr))

	r2_arr = np.zeros(len(t_arr))
	theta2_arr = np.zeros(len(t_arr))

	r1_arr[0] = arm1.r
	theta1_arr[0] = arm1.theta
	r2_arr[0] = arm2.r
	theta2_arr[0] = arm2.theta

	for i,t in enumerate(t_arr[0:-1]):
		v_des = [np.sin(2*np.pi*t/tf), np.cos(2*np.pi*t/tf)]

		arm1.control(v_des,dt)
		arm2.control(v_des,dt)

		r1_arr[i+1] = arm1.r
		theta1_arr[i+1] = arm1.theta

		r2_arr[i+1] = arm2.r
		theta2_arr[i+1] = arm2.theta

	pos1 = np.array([r1_arr*np.cos(theta1_arr),r1_arr*np.sin(theta1_arr)])
	pos2 = np.array([r2_arr*np.cos(theta2_arr)+arm2.base[0],r2_arr*np.sin(theta2_arr)+arm2.base[1]])

	err = npl.norm(pos1-pos2,axis=0)/np.minimum(r1_arr,r2_arr)
	print max(err)


if __name__ == "__main__":
	main()
