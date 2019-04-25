#!/usr/bin/env python3
from lammps import lammps
import numpy as np

def post_force_callback(lammps_ptr, vflag):
  lmp = lammps(ptr=lammps_ptr)
  f = lmp.extract_atom("f", 3) # extract force; 3 for array_double
  df = 50                      # add deviation  
  natoms = lmp.get_natoms()
  for i in range(natoms):
    f[i][0] += df              # f[aid][3]; 0 for x, 1 for y, 2 for z
  
  

