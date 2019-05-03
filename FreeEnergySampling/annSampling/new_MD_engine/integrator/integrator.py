#!/usr/bin/env python3 
from sys import argv
from render import rendering
from customMathFunc import paddingRightMostBin
import numpy as np

def integrator(ndims, cv, force, half_boxboundary, outputfile):

  cv    = np.array(cv)
  force = np.array(force)

  if ndims == 1: # OK

    intg_interval = abs(cv[0] - cv[1])
    accIntg       = 0
    FE            = np.zeros(force.shape[0]) 
    factor        = intg_interval * 0.5 

    for i in range(len(force) - 1):
      accIntg -= (force[i] + force[i+1])  # freeE = -Sf(x)dx
      FE[i] = accIntg * factor 

    FE = paddingRightMostBin(FE)

    with open(outputfile, "w") as fout:
      for i, j in zip(cv, FE):
        fout.write(str(i) + " " + str(j) + "\n")

#---------------------------------------------------------------------#

  if ndims == 2: # probably OK? 

    intg_interval = abs(cv[1][0][0] - cv[1][0][1])
    factor   = intg_interval * 0.5 
    FE_X     = 0
    FE_Y     = 0
    acc_at_x = np.zeros((force.shape[1]))
    acc_at_y = np.zeros((force.shape[1], force.shape[2]))
    FE       = np.zeros((force.shape[1], force.shape[2]))

    for i in range(force.shape[1] - 1):
      FE_X = (force[0][i][0] + force[0][i+1][0])
      acc_at_x[i] -= FE_X 
      for j in range(force.shape[2]- 1):
        FE_Y -= (force[1][i][j] + force[1][i][j+1])
        acc_at_y[i][j] = FE_Y
      FE_Y = 0

    acc_at_x = np.append(acc_at_x, acc_at_x[-1])

    for i in range(force.shape[1]):
      for j in range(force.shape[2]):
        FE[i][j] = (acc_at_x[i] + acc_at_y[i][j]) * factor

    FE += 29.5 
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
    half_boxboundary = np.pi
    with open("force_1D.dat", "r") as fin:
      for line in fin:
        line = line.split() 
        cv.append(float(line[0]))
        force.append(float(line[1]))

  if ndims == 2:
    dsz   = 41 
    cv    = np.zeros((ndims, dsz, dsz))
    force = np.zeros((ndims, dsz, dsz))
    i     = 0
    j     = 0
    half_boxboundary = 2
    with open("force_2D.dat", "r") as fin:
      for line in fin:
        line = line.split()
        cv[0][i][j] = float(line[0])
        cv[1][i][j] = float(line[1])
        force[0][i][j] = float(line[2])
        force[1][i][j] = float(line[3])
        j += 1
        if j == dsz:
          i += 1
          j = 0

  integrator(ndims, cv, force, half_boxboundary,  "FreeE_" + str(ndims) + "D.dat")
