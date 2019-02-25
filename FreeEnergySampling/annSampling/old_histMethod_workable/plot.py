#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-np.pi, np.pi, 361)
y = np.cos(x) + np.cos(2*x) + np.cos(3*x)

with open("estimateU", "w") as fout:
	for i in range(len(x)):
		fout.write(str(x[i]) + " " + str(y[i]) + "\n")
