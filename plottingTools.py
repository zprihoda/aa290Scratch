import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

def animateArm2D(r_arr,theta_arr):
	fig, ax = plt.subplots()
	ax.set(xlim=[0,50],ylim=[-25,25])

	# initialize lines
	arm_line = ax.plot([0,r_arr[0]*np.cos(theta_arr[0])], [0,r_arr[0]*np.sin(theta_arr[0])])[0]
	end_line = ax.plot(r_arr[0]*np.cos(theta_arr[0]), r_arr[0]*np.sin(theta_arr[0]),'.')[0]

	def animate(i):
		x = r_arr[i]*np.cos(theta_arr[i])
		y = r_arr[i]*np.sin(theta_arr[i])
		arm_line.set_xdata([0,x])
		arm_line.set_ydata([0,y])
		end_line.set_xdata(x)
		end_line.set_ydata(y)

	anim = ani.FuncAnimation(fig,animate,interval=5,frames=np.arange(len(r_arr)))
	plt.draw()
	plt.show()
