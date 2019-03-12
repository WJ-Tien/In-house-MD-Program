#!/usr/bin/env python3
import numpy as np
from sympy import *

# TODO FreeE2D

def myRound(a):
	if (a - np.floor(a)) < 0.5:
		return np.floor(a)
	else:
		return np.ceil(a)

def getIndices(input_var, bins):
	binw       = (bins[-1] - bins[0])/ (bins.shape[0] - 1)
	shiftValue = int(myRound(abs(bins[0]) / binw))
	return int(np.floor(input_var/ binw)) + shiftValue

def truncateFloat(f, n=6):
		if f >= 0:
			return np.floor(f * 10 ** n) / 10 ** n
		else:
			return -np.floor(abs(f) * 10 ** n) / 10 ** n

def partitionFunc1D(a, temperature):    # canonical partition function: exp(-U/kbT) / sigma(exp(-U/kbT))
	return np.exp(-(np.cos(a) + np.cos(2*a) + np.cos(3*a))/temperature)	

def partitionFunc2D(a, b, temperature): # canonical partition function: exp(-U/kbT) / sigma(exp(-U/kbT))
	x, y = symbols("x y")	
	Q = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2)) / temperature))  
	Q = lambdify([x, y], Q, "numpy")
	return Q(a, b) 

def boltz1D(a, temperature): # return probability
	q  = partitionFunc1D(a, temperature)
	q  = q.sum(axis=0)
	print(q)
	return np.exp(-(np.cos(a) + np.cos(2*a) + np.cos(3*a))/temperature)/q

def freeE1D(a, temperature):
	p = boltz1D(a, temperature)	
	return -1*temperature*np.log(p)

def boltz2D(a, b, temperature): # exp(-(K+U)/kbT) ~= exp(-K/kbT)exp(-U/kbT) ~= exp(-2/2) * exp(-U/kbT)
	q  = partitionFunc2D(a, b, temperature)
	q  = q.sum(axis=1)
	q  = q.sum(axis=0)
	print(q)
	x, y = symbols("x y")	
	fb = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2)) / temperature))  
	fb = lambdify([x, y], fb, "numpy")
	return fb(a, b) / q

def Usurface1D(a):
	return np.cos(a) + np.cos(2*a) + np.cos(3*a) 
	
def Usurface2D(a, b):
	x, y = symbols("x y")	
	fU = sympify((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2))  
	fU = lambdify([x, y], fU, "numpy")
	return fU(a, b) 

def forcex1D(a):
	return np.sin(a) + 2*np.sin(2*a) + 3*np.sin(3*a) 

def forcex2D(a, b):
	x, y = symbols("x y")	
	fx = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), x)) 
	fx = lambdify([x,y], fx, "numpy")
	return -fx(a, b) 

def forcey2D(a, b):
	x, y = symbols("x y")	
	fy = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), y)) 
	fy = lambdify([x,y], fy, "numpy")
	return -fy(a, b) 

if __name__ == "__main__":
	#pass
	bins = np.linspace(-np.pi, np.pi, 361)
	freeE = freeE1D(bins, 5)
	flat = np.ones(361) * 0.0027777 
	biasPot = -freeE
	flat *= flat*np.exp(biasPot/1/5)
	partitionFunc = np.sum(flat)
	G = -1*5*np.log(flat/partitionFunc)
	#with open("FreeE_1D_T5.dat", "w") as fout:
	with open("unbias.dat", "w") as fout:
		for b, f in zip(bins, G):
			fout.write(str(b) + " " + str(f) + "\n")


