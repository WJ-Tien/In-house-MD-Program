#!/usr/bin/env python3
from lammps import lammps
import numpy as np
import sys
import os

curdir = os.getcwd()
sys.path.append(curdir) 
#from annCore import trainingANN

def post_force_callback(lammps_ptr, vflag):

  lmp         = lammps(ptr=lammps_ptr)

  annCV       = open("annColvars.dat", "a")

  step        = lmp.extract_global("ntimestep", 0)
  natoms      = lmp.get_natoms()
  binSize     = 84 + 1
  lowerBound  = -4.2
  upperBound  = 4.2
  cvRange     = np.linspace(lowerBound, upperBound, binSize) 
  num_coords  = 2                                                  # number of coordinate == target atoms? 
  fb          = lmp.extract_fix("CV", 2, 1)                        # extract biasing force 
  #fb          = np.ctypeslib.as_array(fb, shape=(3*num_coords,))   # each num_coords with 3-dims
  print(fb[0], fb[1], fb[2], fb[3], fb[4])

  """
  # start of training
  output = trainingANN("loss.dat", "hyperparam.dat", 1, nbin)      # 1 for 1D array 

  if step < 150000:
    fb = output.training(cvRange, fb, 0.0058, 0.00005, 20000, 100) # feature, label, learning_rate, epoch, loss-output-freq

  else:
    fb = output.training(cvRange, fb, 0.0058, 0.00000, 20000, 100)

  annCV.write("# %d" % (step) + "\n") 
  for i, j in zip(energyRange, fb) :
    annCV.write(str(i) + " " + str(j) + "\n")
  annCV.write("\n")
  annCV.close()
  # end of training
  """
