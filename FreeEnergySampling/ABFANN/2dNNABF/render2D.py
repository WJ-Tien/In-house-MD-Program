#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

# This code aims at rendering 2d contour plots for 2d langevin toy model

class render2D(object):

	def __init__(self, binNum, half_boundary, temperature):

		self.binNum = binNum 
		self.half_boundary = half_boundary 
		self.binw = 2*self.half_boundary/self.binNum
		self.temperature = temperature 
		self.x = np.arange(-self.half_boundary, self.half_boundary , self.binw)
		self.y = np.arange(-self.half_boundary, self.half_boundary , self.binw)
		self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")


	def partitionFunc(self, a, b): # canonical partition function: exp(-U/kbT) / sigma(exp(-U/kbT))
		x, y = symbols("x y")	
		Q = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2)) / self.temperature))  
		Q = lambdify([x, y], Q, "numpy")
		return Q(a, b) 
	
	def boltz(self, a, b): # exp(-(K+U)/kbT) ~= exp(-K/kbT)exp(-U/kbT) ~= exp(-2/2) * exp(-U/kbT)
		q  = self.partitionFunc(a, b)
		q  = q.sum(axis=0)
		q  = q.sum(axis=0)
		q *= np.exp(-1) # exp(-2/2) for 2d system
		print(q)

		x, y = symbols("x y")	
		fb = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2)) / self.temperature))  
		fb = lambdify([x, y], fb, "numpy")
		return fb(a,b) / q

	def Usurface(self, a, b):
		x, y = symbols("x y")	
		fU = sympify((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2))  
		fU = lambdify([x, y], fU, "numpy")
		return fU(a, b) 

	def forcex(self, a, b):
		x, y = symbols("x y")	
		fx = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), x)) 
		fx = lambdify([x,y], fx, "numpy")
		return fx(a, b) 

	def forcey(self, a, b):
		x, y = symbols("x y")	
		fy = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), y)) 
		fy = lambdify([x,y], fy, "numpy")
		return fy(a, b) 


	def render(self, a, b, func):
		plt.contourf(a, b, func(a, b), 6, alpha=.75, cmap=plt.cm.hot)
		R = plt.contour(a, b, func(a, b), 6, colors='black', linewidth=.5)
		plt.clabel(R, inline=True, fontsize=10)
		plt.xlim(self.x[0],self.x[-1])
		plt.ylim(self.y[0],self.y[-1])
		plt.xticks(np.arange(-self.half_boundary, self.half_boundary, self.binw*4))
		plt.yticks(np.arange(-self.half_boundary, self.half_boundary, self.binw*4))
		plt.savefig(func.__name__ + ".png")
		plt.gcf().clear()

if __name__ == "__main__":

	s = render2D(binNum=40, half_boundary=3, temperature=0.01)
	s.render(s.X, s.Y, s.boltz)
	s.render(s.X, s.Y, s.Usurface)
	s.render(s.X, s.Y, s.forcex)
	s.render(s.X, s.Y, s.forcey)
