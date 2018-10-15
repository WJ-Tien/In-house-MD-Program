#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#https://morvanzhou.github.io/tutorials/data-manipulation/plt/5-1-animation/

def readFile(x):
	with open(x, "r") as fin:
		traj_X = []
		for line in fin:
			line = line.split()
			traj_X.append(float(line[3]))
	return traj_X

def cosPES(x):
	return np.cos(x) + np.cos(2*x) + np.cos(3*x)

PES_X = np.linspace(-np.pi, np.pi, 100)
PES_Y = np.copy(cosPES(PES_X))

traj_X = np.array(readFile(sys.argv[1]))
traj_Y = np.copy(cosPES(traj_X)) #deep copy

fig, ax = plt.subplots(1,1)
dot, = ax.plot([],[],'o', color = 'red')

ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([min(traj_Y), max(traj_Y)+0.1])

def movement(x):
	dot.set_data(traj_X[x], traj_Y[x])

ax.plot(PES_X, PES_Y) #plot PES
a = animation.FuncAnimation(fig, movement, frames=len(traj_Y), interval=1) #plot the paritcle traj
a.save(sys.argv[2], fps=int(sys.argv[3]), extra_args=['-vcodec', 'libx264'])
plt.show()

