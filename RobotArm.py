import numpy as np
import numpy.linalg as npl


class RobotArm1D():
	def __init__(self,r,theta,x0=0,y0=0):
		self.r = r
		self.theta = theta
		self.base = np.array([x0,y0])

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
