#!/usr/bin/env python3
import sys
import numpy as np

# p(x) ~ exp(-(K+U)/kbT) ~= A*exp(-U(x) / kbT)
# U(x) = cos(x) + cos(2*x) + cos(3*x)

with open(sys.argv[1], "r") as fin:
	coord = [] 
	for line in fin:
		line = line.split()
		coord.append(float(line[2])) # save system coordinates
	coord.sort()
	nsamples = len(coord) #total number of samples in the system
			
binw = float(sys.argv[2]) # bin width
minb = int(np.floor(coord[0] / binw))  # lower boudary
maxb = int(np.floor(coord[-1] / binw)) # upper boundary

prob = [0] * (maxb - minb + 1) # initialization of probalitity array 

for i in range(len(coord)):
	prob[int(np.floor(coord[i] / binw)) + abs(minb)] += 1 # shift the index with value abs(minb)

prob = np.array(prob)
prob = (prob / nsamples) #final probability distribution 
x_axis = np.arange(coord[0], coord[-1] + binw, binw) #initialization of x-axis

with open(sys.argv[3], "w") as fout:
	for i in range(len(prob)):
		fout.write(str(x_axis[i]) + " " + str(prob[i]) + "\n")
