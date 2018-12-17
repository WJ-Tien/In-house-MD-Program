#!/usr/bin/env python3

import numpy as np

binNum = 360
binw = 2*np.pi / binNum 
x = np.arange(-np.pi, np.pi + binw, binw)
y = np.sin(x) + 2*np.sin(2*x) + 3*np.sin(3*x)
print(max(y))
'''
with open("estimate", "w") as fout:
	for i in range(binNum):
		fout.write(str(x[i]) + " " +  str(y[i]) + "\n")
'''
