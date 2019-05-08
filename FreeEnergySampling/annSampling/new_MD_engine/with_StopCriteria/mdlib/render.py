#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mdlib.customMathFunc import forcex2D, forcey2D, Usurface2D, boltz2D, Usurface1D, forcex1D, boltz1D, freeE1D, freeE2D

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
  pass
  #s = rendering(ndims=2, half_boxboundary=3, binNum=41, temperature=0.1)
  #s.render(boltz2D, name="boltz2D")
  #s = rendering(ndims=2, half_boxboundary=2, binNum=41, temperature=0.001)
  #s.render(boltz2D, name="boltz2D")
  #s = rendering(ndims=2, half_boxboundary=2.5, binNum=33)
  #s.render(forcex2D  ,name="forcex2D")
  #s.render(forcey2D  ,name="forcey2D")
  #s = rendering(ndims=2, half_boxboundary=2.0, binNum=40)
  #s.render(forcex2D  ,name="forcex2D")
  #s.render(forcey2D  ,name="forcey2D")
  #s = rendering(ndims=1, half_boxboundary=np.pi, binNum=361, temperature=0.75)
  #s.render(freeE1D, name="freeE1D" )
  #s.render(boltz1D, name="boltz1D")
  #s = rendering(ndims=1, half_boxboundary=np.pi, binNum=361)
  #s.render(forcex1D  ,name="forcex1D")
  #s = rendering(ndims=2, half_boxboundary=2, binNum=40, temperature=4)
  #s.render(freeE2D, name="freeE2D")

