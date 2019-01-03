#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

'''
def f(a, b):
	x, y = symbols("x y")	
	
	fx =  sympify(diff(1.00*exp(-0.5*(((x+1.000) / 1.0)**2 + ((y+20.00) / 2.3)**2)) +\
                     2.00*exp(-0.5*(((x-7.231) / 1.4)**2 + ((y-3.000) / 0.5)**2)) +\
                     1.20*exp(-0.5*(((x-0.111) / 8.5)**2 + ((y+1.000) / 7.5)**2)) +\
                     0.03*exp(-0.5*(((x+0.000) / 1.2)**2 + ((y+4.000) / 0.5)**2)) +\
                     7.00*exp(-0.5*(((x-1.812) / 5.0)**2 + ((y-0.568) / 0.5)**2)) +\
                     0.01*exp(-0.5*(((x+7.500) / 1.3)**2 + ((y+4.123) / 0.5)**2)) +\
                    0.005*exp(-0.5*(((x-8.600) / 2.5)**2 + ((y-4.000) / 0.5)**2)) +\
                     1.50*exp(-0.5*(((x+5.000) / 3.1)**2 + ((y-0.440) / 0.1)**2)) +\
                     0.05*exp(-0.5*(((x-0.001) / 1.5)**2 + ((y+0.777) / 0.9)**2)) +\
                    10.50*exp(-0.5*(((x+0.007) / 1.7)**2 + ((y-0.987) / 2.5)**2)), x))
	
	fx = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), x)) 
	fx = lambdify([x,y], fx, "numpy")
	return fx(a, b) 
'''

def boltz(a, b):
	x, y = symbols("x y")	
	fs = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2))/10))  
	fs = lambdify([x, y], fs, "numpy")
	return fs(a, b) 

def forcex(a, b):
	x, y = symbols("x y")	
	fx = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), x)) 
	fx = lambdify([x,y], fx, "numpy")
	return fx(a, b) 

def forcey(a, b):
	x, y = symbols("x y")	
	fy = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), y)) 
	fy = lambdify([x,y], fy, "numpy")
	return fy(a, b) 

x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X,Y = np.meshgrid(x, y)
plt.contourf(X, Y, forcex(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y, forcex(X, Y), 8, colors='black', linewidth=.5)
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
#plt.show()
#plt.savefig("expUxy.png")
plt.savefig("forcex.png")


