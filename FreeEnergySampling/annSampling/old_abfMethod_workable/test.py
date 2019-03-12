#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def z(x,y):
	return 3*(1-x)**2 * np.exp(-x**2-(y+1)**2) - 10*(x/5-x**3-y/5)*np.exp(-x**2-y**2)- (0.333*np.exp(-(x+1)**2-y**2))

x = np.linspace(-3 ,3, 50)
y = np.linspace(-3, 3, 50)
A, B = np.meshgrid(x, y, indexing="ij")
cs = plt.contourf(A, B, z(A, B), 8, cmap=plt.cm.plasma)
R  = plt.contour(A, B, z(A, B), 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
plt.clabel(R, inline=True, fontsize=8)
plt.colorbar(cs)
plt.show()
