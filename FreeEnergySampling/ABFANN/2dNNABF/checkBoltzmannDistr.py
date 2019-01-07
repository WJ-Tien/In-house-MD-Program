#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

# p(x) ~ exp(-(K+U)/kbT) ~= A*exp(-U(x) / kbT)
# Apprantly, the kinetic energy term is constant when the temperature is constant
# U(x) = cos(x) + cos(2*x) + cos(3*x)
# binNum +1 to prevent boundary errors 
ndims            = int(sys.argv[1])
half_boxboundary = float(sys.argv[2])
binNum           = int((sys.argv[3]))
binw             = 2 * half_boxboundary / binNum  
abfcheckflag     = sys.argv[6]
nncheckflag      = sys.argv[7]

with open(sys.argv[4], "r") as fin:

	if ndims == 1: 
		prob_x = np.zeros((binNum+1), dtype = np.int32)  
		coord_x = []  
		for line in fin:
			line = line.split()
			if line[0] != "#":	
				coord_x.append(round(float(line[2]), 7)) # 2 for cartcoord_1D

		nsamples = len(coord_x) 

		for i in range(len(coord_x)):
			prob_x[int(np.floor(coord_x[i] / binw)) + binNum//2] += 1 # shift the index to prevent the negative index error
			
		prob_x = np.array(prob_x)
		prob_x = (prob_x / nsamples) # final probability distribution 
		x_axis = np.arange(-half_boxboundary, half_boxboundary, binw) # initialization of x-axis

		with open(sys.argv[5], "w") as fout:
			for i in range(len(prob_x)-1): # discard the point on pi (PBC issues)
				fout.write(str(x_axis[i]) + " " + str(prob_x[i]) + "\n")


	if ndims == 2:

		prob_xy  = np.zeros((binNum+1, binNum+1), dtype = np.float32)  
		coord_x = [] 
		coord_y = [] 

		for line in fin:
			line = line.split()
			if line[0] != "#":	
				coord_x.append(round(float(line[2]), 7)) # 2 for cartcoord_1D, 2 4 for cartcoord_2D
				coord_y.append(round(float(line[4]), 7)) # 2 for cartcoord_1D, 2 4 for cartcoord_2D

		nsamples = len(coord_x) 
		

		for i in range(len(coord_x)):
			prob_xy[int(np.floor(coord_x[i] / binw)) + binNum//2][int(np.floor(coord_y[i] / binw)) + binNum//2] += 1 # shift the index with value abs(minb)

		x_axis = np.arange(-half_boxboundary, half_boxboundary, binw) # initialization of x-axis
		y_axis = np.arange(-half_boxboundary, half_boxboundary, binw) # initialization of x-axis

		prob_xy = (prob_xy / nsamples) # final probability distribution 

		with open(sys.argv[5], "w") as fout:
			for i in range(prob_xy.shape[0]-1): # discard the point on the positive boundary (PBC issues)
				for j in range(prob_xy.shape[0]-1): # discard the point on the positive boundary (PBC issues)
					fout.write(str(x_axis[i]) + " ")
					fout.write(str(y_axis[j]) + " " +  str(prob_xy[i][j]) + "\n")

		prob_xy = np.delete(prob_xy, -1, 0) # To equalize x & y & prob_xy array size
		prob_xy = np.delete(prob_xy, -1, 1)

		X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")

		plt.contourf(X, Y, prob_xy, 6, alpha=.75, cmap=plt.cm.hot)
		C = plt.contour(X, Y, prob_xy, 6, colors='black', linewidth=.5)
		plt.clabel(C, inline=True, fontsize=10)
		plt.xlim(x_axis[0],x_axis[-1])
		plt.ylim(y_axis[0],y_axis[-1])
		plt.xticks(np.arange(-half_boxboundary, half_boxboundary, binw*4))
		plt.yticks(np.arange(-half_boxboundary, half_boxboundary, binw*4))
		plt.savefig(abfcheckflag + "_" + nncheckflag + "_" + "boltz2d.png")	

