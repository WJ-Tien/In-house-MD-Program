#!/usr/bin/env python3
from lammps import lammps
import numpy as np
import sys
import os

curdir = os.getcwd()
sys.path.append(curdir) #TODO add path
from annCore import trainingANN

def post_force_callback(lammps_ptr, vflag):

	lmp = lammps(ptr=lammps_ptr)

	step        = lmp.extract_global("ntimestep", 0)
	natoms      = lmp.get_natoms()
	Ehigh       = -4.9
	Elow        = -6.2
	binsize     = 2.0
	nbin        = int(np.ceil((Ehigh-Elow) * natoms / binsize)) + 1
	energyRange = np.linspace(-6.2, -4.9, nbin) 

	T = lmp.extract_fix("ST", 2, 1) # 221 222 200 works
	T = np.ctypeslib.as_array(T, shape=(nbin,))  #binNum + 1
	T[-1] = T[0] #padding  rightMostBin 

	# training ~~
	output = trainingANN("loss.dat", "hyperparam.dat", 1, nbin) # binNum + 1; therefore 72 + 1 
	if step < 1000000:
		T = output.training(energyRange, T, 0.025, 0.00025, 2500, 100)
	else:
		T = output.training(energyRange, T, 0.025, 0.00025, 7500, 100)
