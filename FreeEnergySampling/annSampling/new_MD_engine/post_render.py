#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# This code aims at rendering 1d x-y plots and 2d contour plots for Langevin toy model

class rendering(object):

  def __init__(self, ndims, half_boxboundary, binNum, temperature=None):
    self.temperature      = temperature
    self.ndims            = ndims
    self.half_boxboundary = half_boxboundary
    self.binNum           = binNum

  def render(self, renderObj, name):

    if self.ndims == 1:

      x_axis = np.linspace(-self.half_boxboundary, self.half_boxboundary, self.binNum+1)

      if hasattr(renderObj, '__call__'): # is function
        if renderObj.__name__ == "boltz1D" or renderObj.__name__ == "freeE1D":
          plt.plot(x_axis, renderObj(x_axis, self.temperature))
        else:
          plt.plot(x_axis, renderObj(x_axis))

      else:                              # is array
          plt(x_axis, renderObj)

      plt.xticks(np.linspace(-self.half_boxboundary, self.half_boxboundary, 8))
      plt.savefig(name + ".png")
      plt.gcf().clear()

    if self.ndims == 2: 

      x_axis = np.linspace(-self.half_boxboundary, self.half_boxboundary, self.binNum+1)
      y_axis = np.linspace(-self.half_boxboundary, self.half_boxboundary, self.binNum+1)

      A, B = np.meshgrid(x_axis, y_axis, indexing="ij")

      if hasattr(renderObj, '__call__'):
        if renderObj.__name__ == "boltz2D" or renderObj.__name__ == "freeE2D": 
          cs = plt.contourf(A, B, renderObj(A, B, self.temperature), 8, cmap=plt.cm.plasma)
          R  = plt.contour(A, B, renderObj(A, B, self.temperature), 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
        else:
          cs = plt.contourf(A, B, renderObj(A, B), 8, cmap=plt.cm.plasma)
          R  = plt.contour(A, B, renderObj(A, B), 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
          
      else: 
        cs = plt.contourf(A, B, renderObj, 8, cmap=plt.cm.plasma)
        R  = plt.contour(A, B, renderObj, 8, colors='black', linewidth=.25, linestyles="solid", extend="both")

      plt.clabel(R, inline=True, fontsize=8)
      plt.xlim(x_axis[0],x_axis[-1])
      plt.ylim(y_axis[0],y_axis[-1])
      plt.xticks(np.linspace(-self.half_boxboundary, self.half_boxboundary, 6))
      plt.yticks(np.linspace(-self.half_boxboundary, self.half_boxboundary, 6))
      plt.colorbar(cs)
      plt.savefig(name + ".png")
      plt.gcf().clear()

if __name__ == "__main__":
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
