#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mdlib.render import rendering

s = rendering(2, 2, 40, 2)
FE = np.zeros((41,41))
Force = np.zeros((2, 41,41))
Hist = np.zeros((41,41)) 

with open("FreeE_m1.0T2.00000_gamma0.1_len_36000000.dat" , "r") as fin:
  i = 0
  j = 0
  for line in fin:
    line = line.split()
    FE[i][j] = float(line[2])
    j += 1
    if j == 41:
      i += 1
      j = 0
    
with open("Hist_m1.0T2.00000_gamma0.1_len_36000000.dat" , "r") as fin:
  i = 0
  j = 0
  for line in fin:
    line = line.split()
    Hist[i][j] = float(line[2])
    j += 1
    if j == 41:
      i += 1
      j = 0

with open("Force_m1.0T2.00000_gamma0.1_len_36000000.dat" , "r") as fin:
  i = 0
  j = 0
  for line in fin:
    line = line.split()
    Force[0][i][j] = float(line[2])
    Force[1][i][j] = float(line[4])
    j += 1
    if j == 41:
      i += 1
      j = 0

s.render(FE, "yes_yes_FreeE")
s.render(Force[0], "yes_yes_ForceX")
s.render(Force[1], "yes_yes_ForceY")
s.render(Hist, "yes_yes_Boltzmann")
