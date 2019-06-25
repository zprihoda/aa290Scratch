import numpy as np
import numpy.linalg as npl

class RobotArm2D():
	def __init__(self,r,theta,base=[0,0]):
		"""
		2D Robotic Arm
		Arguments:
			r: initial length of arm
			theta: rotation angle (measured from counter-clockwise from x)
			base: specifies where the base of the arm is attached
		"""
		self.r = r
		self.theta = theta
		self.base = np.array(base)

	def pos(self):
		x = self.r*np.cos(self.theta) + self.base[0]
		y = self.r*np.sin(self.theta) + self.base[1]
		return np.array([x,y])

	def control(self,v,dt):

		r = self.r
		theta = self.theta

		B = np.array([[np.cos(theta), -r*np.sin(theta)],
					  [np.sin(theta), r*np.cos(theta)]])

		u = npl.solve(B,v)
		r_dot = u[0]
		theta_dot = u[1]

		self.r = r + r_dot*dt
		self.theta = theta + theta_dot*dt
		return r_dot, theta_dot
