import numpy as np
import numpy.linalg as npl

GRID_SIZE = 1.

grid1 = np.array([
	[1,1,1,1,1,1,1,1,1,1],
	[1,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,0,0,0,0,0,1],
	[1,0,0,0,0,0,0,0,0,1],
	[1,0,0,1,1,1,1,0,0,1],
	[1,0,0,1,0,0,1,0,0,1],
	[1,0,0,1,0,0,1,0,0,1],
	[1,0,0,1,0,0,1,0,0,1],
	[1,1,1,1,0,0,1,1,1,1]
	])

def getEdges(grid):
	i_arr, j_arr = np.where(grid==1)
	pos_arr = [np.zeros(2)]*len(i_arr)
	for k in range(len(i_arr)):
		pos_arr[k] = idx2pos(i_arr[k],j_arr[k])
	return np.array(pos_arr)

def idx2pos(i,j):
	""" Convert from array idx to x,y coordinates """
	x = j*GRID_SIZE
	y = (len(grid1)-1-i)*GRID_SIZE
	return np.array([x,y])

def pos2idx(pos):
	""" Return the nearest grid point to the given position"""
	j = int(np.round(pos[0]/GRID_SIZE))
	i = int(((len(grid1)-1) - np.round(pos[1]))/GRID_SIZE)
	return i,j

def lineIntersectsGrid(p1,p2,grid):
	unit_vec = (p2-p1)/npl.norm(p2-p1)
	dist = npl.norm(p2-p1)

	line = p1[:,np.newaxis] + np.arange(0,dist,GRID_SIZE/2)*unit_vec[:,np.newaxis]

	for i in range(line.shape[1]):
		pos = line[:,i]
		idx = pos2idx(pos)
		if grid[idx] == 1:
			return True
	return False
