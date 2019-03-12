#!/usr/bin/env python3
from lammps import lammps
import numpy as np
import sys

#import sys
#import os
#curdir = os.getcwd()
#sys.path.append(curdir) #TODO add path
#from train import test

def post_force_callback(lammps_ptr, vflag):

	lmp = lammps(ptr=lammps_ptr)
	natoms = lmp.get_natoms()
	#print(natoms)
	#v = lmp.extract_variable("u", "all", 1) # 221 222 200 works
	step = lmp.extract_global("ntimestep", 0)
	f3 = lmp.extract_fix("3", 2, 1) # 221 222 200 works
	print(f3[0])
	#print(f3)
	if step == 5000:
		f3[0] = 100000000
	#print(f3[0])

	#f4 = lmp.extract_fix("4", 1, 2) # 221 222 200 works
	#print(f2[0][0], f2[0][1], f2[0][2])
	#print(f4[0][0], f4[0][1], f4[0][2])
	#print(v[0])
	#f2 = np.ctypeslib.as_array(f2.contents, shape=(natoms, 3)) 
	#print(f2)

	# workable
	#f = lmp.extract_atom("f", 3) 
	#F = np.ctypeslib.as_array(f.contents,shape=(natoms, 3)) # c_double -> np.ndarray
	#print(sys.getsizeof(F))
	#test()	
	#df = 50
	#for i in range(natoms):
	#	F[i][0] += df							 
		

# https://stackoverflow.com/questions/24605464/how-to-access-pointer-to-pointer-values-in-python
# https://docs.scipy.org/doc/numpy/reference/routines.ctypeslib.html
# lammps.LP_LP_c_double --> np.ndarray by np.ctypeslib.as_array(obj=f.contents,shape=(natoms,3))
# numpy ndarray shares the same memory blocks with ctypes pointer
