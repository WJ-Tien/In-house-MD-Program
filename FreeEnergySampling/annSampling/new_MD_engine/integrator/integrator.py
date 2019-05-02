#!/usr/bin/env python3 
from sys import argv
from render import rendering
from customMathFunc import paddingRightMostBin
import numpy as np

def integrator(ndims, cv ,force, half_boxboundary, outputfile):

  if ndims == 1:

    accIntg  = 0
    FE       = []
    interval = abs(cv[0] - cv[1])
    factor   = (interval * 0.5) ** ndims

    for i in range(len(force) - 1):
      accIntg -= (force[i] + force[i+1]) 
      FE.append(accIntg * factor)
    
    FE = paddingRightMostBin(FE)

    with open(outputfile, "w") as fout:
      for i, j in zip(cv, FE):
        fout.write(str(i) + " " + str(j) + "\n")

  #--------------------------------------------------#

  if ndims == 2:

    interval = abs(cv[1][0][0] - cv[1][0][1])
    factor   = (interval * 0.5) ** ndims 
    FE_X     = 0
    FE_Y     = 0
    acc_at_x = np.zeros((force.shape[1], force.shape[2]))
    acc_at_y = np.zeros((force.shape[1], force.shape[2]))
    FE       = np.zeros((force.shape[1], force.shape[2]))

    for i in range(force.shape[1] - 1):
      for j in range(force.shape[2]- 1):
        FE_X = (force[0][i][j] + force[0][i + 1][j])
        acc_at_x[i][j] -= FE_X 
        FE_Y -= (force[1][i][j] + force[1][i][j+1])
        acc_at_y[i][j] = FE_Y
      FE_Y = 0

    for i in range(force.shape[1]):
      for j in range(force.shape[2]):
        FE[i][j] = (acc_at_x[i][j] + acc_at_y[i][j]) * factor

    FE += 30
    FE = paddingRightMostBin(FE)

    with open(outputfile, "w") as fout:
      for i in range(force.shape[1]):
        for j in range(force.shape[2]):
          fout.write(str(cv[0][i][j]) + " " + str(cv[1][i][j]) + " " + str(FE[i][j]) +  "\n")

    s = rendering(ndims, half_boxboundary, force.shape[1] - 1)
    s.render(FE, name="FreeE_2D")
    
if __name__ == "__main__":

  ndims = int(argv[1])

  if ndims == 1:
    cv    = []
    force = []
    with open(argv[2], "r") as fin:
      for line in fin:
        line = line.split() 
        cv.append(float(line[0]))
        force.append(float(line[1]))

  if ndims == 2:
    cv    = np.zeros((2, 41, 41))
    force = np.zeros((2, 41, 41))
    i     = 0
    j     = 0
    with open(argv[2], "r") as fin:
      for line in fin:
        line = line.split()
        cv[0][i][j] = float(line[0])
        cv[1][i][j] = float(line[1])
        force[0][i][j] = float(line[2])
        force[1][i][j] = float(line[3])
        j += 1
        if j == 41:
          i += 1
          j = 0

  #integrator(ndims, cv, force, np.pi,  "FreeE_" + str(ndims) + "D.dat")
  integrator(ndims, cv, force, 2,  "FreeE_" + str(ndims) + "D.dat")
