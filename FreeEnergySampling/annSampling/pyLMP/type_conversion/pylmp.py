#!/usr/bin/env python3
from lammps import lammps
import numpy as np
from ctypes import c_double

def post_force_callback(lammps_ptr, vflag):
	lmp = lammps(ptr=lammps_ptr)
	natoms = lmp.get_natoms()
	f = lmp.extract_atom("f", 3) 
	df = 500                      
	F = np.ctypeslib.as_array(f.contents,shape=(natoms,3)) # c_double -> np.ndarray
	for i in range(natoms):
		F[i][0] += df              
		
	


# lammps.LP_LP_c_double --> np.ndarray by np.ctypeslib.as_array(obj=f.contents,shape=(natoms,3))
# numpy ndarray shares the same memory blocks with ctypes pointer

# learn force using np nd array
	

