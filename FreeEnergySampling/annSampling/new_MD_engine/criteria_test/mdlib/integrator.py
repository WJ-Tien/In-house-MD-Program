#!/usr/bin/env python3 
from mdlib.render import rendering
from mdlib.customMathFunc import paddingRightMostBin
import numpy as np

def integrator(ndims, cv, force, half_boxboundary, frame, shiftConst, outputfile):

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

    FE += shiftConst 
    FE = paddingRightMostBin(FE)

    with open(outputfile, "a") as fout:
      fout.write("#" + " " + str(frame) + "\n")
      for i, j in zip(cv, FE):
        fout.write(str(i) + " " + str(j) + "\n")
      fout.write("\n")


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

    #FE += 29.5 
    FE += shiftConst 
    FE = paddingRightMostBin(FE)

    with open(outputfile, "a") as fout:
      fout.write("#" + " " + str(frame) + "\n")
      for i in range(force.shape[1]):
        for j in range(force.shape[2]):
          fout.write(str(cv[0][i][j]) + " " + str(cv[1][i][j]) + " " + str(FE[i][j]) +  "\n")
      fout.write("\n")

    s = rendering(ndims, half_boxboundary, force.shape[1] - 1)
    s.render(FE, name="FreeE_2D")

  return FE
    
if __name__ == "__main__":
  pass
  
