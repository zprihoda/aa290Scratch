import numpy as np
import matplotlib.pyplot as plt

import edgeGrid
import plottingTools

MAX_REACH = 4

def main():
	grid = edgeGrid.grid1

	start_pos = np.array([1,1])
	end_pos = np.array([8,1])

	print edgeGrid.lineIntersectsGrid(start_pos, end_pos, grid)
	print edgeGrid.lineIntersectsGrid(start_pos, np.array([1,8]), grid)

	# fig,ax = plt.subplots()
	# ax.plot(start_pos[0],start_pos[1],'b.')
	# ax.plot(end_pos[0],end_pos[1],'rx')
	# plottingTools.plotGrid(grid, ax=ax)
	# plottingTools.plotCircle(end_pos, MAX_REACH, ax=ax)
	# plt.show()


if __name__ == "__main__":
	 main()
