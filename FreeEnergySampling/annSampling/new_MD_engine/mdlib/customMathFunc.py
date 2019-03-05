#!/usr/bin/env python3
import numpy as np

# TODO FreeE2D

def randMars():
  # https://github.com/lammps/lammps/blob/master/src/random_mars.cpp
	seed = np.random.randint(0, 900000000) 
	save = 0
	u = np.zeros(98)
	ij = (seed-1) / 30082
	kl = (seed-1) - 30082*ij
	i = (ij/177) % 177 + 2
	j = ij %177 + 2
	k = (kl/169) % 178 + 1
	l = kl % 169
	for ii in range(1, 98):	
		s = 0.0
		t = 0.5
		for jj in range(1,25):
			m = ((i*j) % 179)*k % 179
			i = j
			j = k
			k = m
			l = (53*l+1) % 169
			if l*m % 64 >= 32:
				s += t
			t *=0.5
		u[ii] = s	

	c = 362436.0 / 16777216.0
	cd = 7654321.0 / 16777216.0
	cm = 16777213.0 / 16777216.0
	i97 = 97
	j97 = 33
	uni = u[i97] - u[j97]
	if uni < 0.0:
		uni += 1
	u[i97] = uni
	i97 -= 1
	if i97 == 0:
		i97 = 97
	j97 -= 1
	if j97 == 0:
		j97 = 97
	c -= cd
	if c < 0.0:
		c += cm
	uni -= c
	if uni < 0.0:
		uni += 1
	return uni

def myRound(a):
	if (a - np.floor(a)) < 0.5:
		return np.floor(a)
	else:
		return np.ceil(a)

def getIndices(input_var, bins):
	binw			 = (bins[-1] - bins[0])/ (bins.shape[0] - 1)
	shiftValue = int(myRound(abs(bins[0]) / binw))
	return int(np.floor(input_var/ binw)) + shiftValue

def paddingRighMostBins(ndims, input_numpy_array):
	""" Detail with the rightmost bin.
			When accumulating the counts on the colvars, we neglect the counts of the rightmost bins since it usually causes some floating point precision issues. 
			This simply originated from the implementation I used, i.e., accumulate the histogram by using left close method e.g. value between 0~1 belongs to 0.
			Hence, I pad the rightmost bin = leftmost bin since we apply PBC condition to the calculation
	"""
	if ndims == 1:
		input_numpy_array[-1] = input_numpy_array[0] 

	if ndims == 2:
		input_numpy_array[0, -1, :] = input_numpy_array[0, 0, :]
		input_numpy_array[0, :, -1] = input_numpy_array[0, :, 0]
		input_numpy_array[1, -1, :] = input_numpy_array[1, 0, :]
		input_numpy_array[1, :, -1] = input_numpy_array[1, :, 0]

	return input_numpy_array

def truncateFloat(f, n=7):
		if f >= 0:
			return np.floor(f * 10 ** n) / 10 ** n
		else:
			return -np.floor(abs(f) * 10 ** n) / 10 ** n

def partitionFunc1D(a, temperature):		# canonical partition function: exp(-U/kbT) / sigma(exp(-U/kbT))
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
	pass
	bins = np.linspace(-np.pi, np.pi, 361)
	
	T = 0.75
	freeE = freeE1D(bins, 0.75)
	with open("FreeE_1D_T%f.dat" %(T), "w") as fout:
		for b, f in zip(bins, freeE):
			fout.write(str(b) + " " + str(f) + "\n")

	#boltz = boltz1D(bins, 0.05)
	#with open("boltz_1D_T0.05.dat", "w") as fout:
	#	for b, f in zip(bins, boltz):
	#		fout.write(str(b) + " " + str(f) + "\n")
	#acc = 0
	#sq = 0
	#n = 100000
	#for i in range(n):
	#	s = randMars() -0.5
	#	acc += s
	#	sq += s**2
	#print(acc/n)
	#print(sq/n)
