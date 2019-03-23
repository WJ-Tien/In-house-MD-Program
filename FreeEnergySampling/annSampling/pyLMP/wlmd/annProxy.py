#!/usr/bin/env python3
from lammps import lammps
import numpy as np
import sys
import os

curdir = os.getcwd()
sys.path.append(curdir) 
from annCore import trainingANN

def post_force_callback(lammps_ptr, vflag):

  lmp = lammps(ptr=lammps_ptr)

  annST       = open("annST.dat", "a")

  step        = lmp.extract_global("ntimestep", 0)
  natoms      = lmp.get_natoms()
  Ehigh       = -4.9
  Elow        = -6.2
  binsize     = 2.0
  nbin        = int(np.ceil((Ehigh-Elow) * natoms / binsize)) 
  energyRange = np.linspace(-6.2, -4.9, nbin) 

  T           = lmp.extract_fix("ST", 2, 1) # extract T(U) 
  T           = np.ctypeslib.as_array(T, shape=(nbin,)) 
  doANN       = lmp.extract_fix("ST", 2, 2) # extract ANN Flag 

  # training
  output = trainingANN("loss.dat", "hyperparam.dat", 1, nbin) # 1 for 1D array 

  if (doANN[0][0]): # doANN[0[0] = 1 when step % (annFreq = 5000000) == 0; 

    if step < 5000000:
      result = output.training(energyRange, T, 0.0058, 0.00005, 20000, 100) # feature, label, learning_rate, epoch, loss-output-freq

    else:
      result = output.training(energyRange, T, 0.0058, 0.00000, 20000, 100)

    annST.write("# STEP %d" % (step) + "\n") 
    for i, j in zip(energyRange, result) :
      annST.write(str(i) + " " + str(j) + "\n")
    annST.write("\n")
  annST.close()


