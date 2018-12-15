#!/usr/bin/env python3
import numpy as np
import os

curdir = os.getcwd()
localHist = [] 
stdev_avg = []
x = 0
xsq = 0

for data in os.listdir(curdir):
	if data.startswith("wABFHistogram"):
		with open(data, "r") as f:
			for line in f:
				localHist.append(float(line.split()[1]))	

		if localHist[0] / localHist[len(localHist)//2] <= 0.01:
			del localHist[0] # discard the largest deviation due to float point error at the boundary
		
		for i in localHist:
			x += i
			xsq += i**2

		stdev_avg.append(np.sqrt((xsq/len(localHist)) - (x/len(localHist))**2))
		localHist = []
		x = 0
		xsq = 0


print(sum(stdev_avg) / len(stdev_avg))

			
		 
