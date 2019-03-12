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
	#natoms = lmp.get_natoms()
	#print(natoms)
	f = lmp.extract_fix("2", 2, 1) # 221 222 200 works
	print(f[0])
	f[0] = 1.0
	#f = np.ctypeslib.as_array(fixx.contents, shape=(natoms, 3)) 
	#print(f)

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
