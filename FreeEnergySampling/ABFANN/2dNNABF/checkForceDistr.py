#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

ndims            = int(sys.argv[1])
half_boxboundary = float(sys.argv[2])
binNum           = int((sys.argv[3]))
binw             = 2 * half_boxboundary/ binNum  #sys.argv[2] =  half_boxboundary, sys.argv[3] = binNum
abfcheckflag     = sys.argv[5]
nncheckflag      = sys.argv[6]

with open(sys.argv[4], "r") as fin:
	
	if ndims == 1:
		exit(0)

	else:
		force_x  = np.zeros((binNum, binNum), dtype=np.float32) 
		force_y  = np.zeros((binNum, binNum), dtype=np.float32) 
		coord_x = [] 
		coord_y = [] 

		for line in fin:
			line = line.split()
			if line[0] != "#":	
				coord_x.append(round(float(line[0]), 7)) # 0 for cartcoord_x 
				coord_y.append(round(float(line[1]), 7)) # 1 for cartcoord_y
				force_x[int(np.floor(round(float(line[0]), 7) / binw)) + binNum//2][int(np.floor(round(float(line[1]), 7) / binw)) + binNum//2] = (round(float(line[2]), 7)) # 2 for force_x
				force_y[int(np.floor(round(float(line[0]), 7) / binw)) + binNum//2][int(np.floor(round(float(line[1]), 7) / binw)) + binNum//2] = (round(float(line[4]), 7)) # 4 for force_y
				
		x_axis = np.arange(-half_boxboundary, half_boxboundary, binw)
		y_axis = np.arange(-half_boxboundary, half_boxboundary, binw)
		X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")

		cs = plt.contourf(X, Y, force_x, 6, alpha=.75, cmap=plt.cm.hot)
		C = plt.contour(X, Y, force_x, 6, colors='black', linewidth=.5)
		plt.clabel(C, inline=True, fontsize=10)
		plt.xlim(x_axis[0],x_axis[-1])
		plt.ylim(y_axis[0],y_axis[-1])
		plt.xticks(np.arange(-half_boxboundary, half_boxboundary, binw*4))
		plt.yticks(np.arange(-half_boxboundary, half_boxboundary, binw*4))
		plt.colorbar(cs)
		plt.savefig(abfcheckflag + "_" + nncheckflag + "_" + "forcex.png")	

		plt.gcf().clear()

		cs = plt.contourf(X, Y, force_y, 6, alpha=.75, cmap=plt.cm.hot)
		D = plt.contour(X, Y, force_y, 6, colors='black', linewidth=.5)
		plt.clabel(D, inline=True, fontsize=10)
		plt.xlim(x_axis[0],x_axis[-1])
		plt.ylim(y_axis[0],y_axis[-1])
		plt.xticks(np.arange(-half_boxboundary, half_boxboundary, binw*4))
		plt.yticks(np.arange(-half_boxboundary, half_boxboundary, binw*4))
		plt.colorbar(cs)
		plt.savefig(abfcheckflag + "_" + nncheckflag + "_" + "forcey.png")	
		




