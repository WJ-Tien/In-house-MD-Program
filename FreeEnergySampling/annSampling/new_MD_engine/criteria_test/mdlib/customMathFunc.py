#!/usr/bin/env python3
import numpy as np
from sympy import *

# TODO 2019/4
def integrator():
  pass

def myRound(a):
  if (a - np.floor(a)) < 0.5:
    return np.floor(a)
  else:
    return np.ceil(a)

def getIndices(input_var, bins):
  bins = np.array(bins)
  binw       = (bins[-1] - bins[0])/ (bins.shape[0] - 1)
  shiftValue = int(myRound(abs(bins[0]) / binw))
  return int(np.floor(input_var/ binw)) + shiftValue

def paddingRightMostBin(input_numpy_array):
  """ Detail with the rightmost bin.
      When accumulating the counts on the colvars, we neglect the counts of the rightmost bins since it usually causes some floating point precision issues. 
      This simply originated from the implementation I used, i.e., accumulate the histogram by using left close method e.g. value between 0~1 belongs to 0.
      Hence, I pad the rightmost bin = leftmost bin since we apply PBC condition to the calculation
  """
  input_numpy_array = np.array(input_numpy_array)

  if len(input_numpy_array.shape) == 1:
    input_numpy_array[-1] = input_numpy_array[0] 

  elif len(input_numpy_array.shape) == 2:
    input_numpy_array[-1, :] = input_numpy_array[0, :]
    input_numpy_array[:, -1] = input_numpy_array[:, 0]

  elif len(input_numpy_array.shape) == 3:
    input_numpy_array[0, -1, :] = input_numpy_array[0, 0, :]
    input_numpy_array[0, :, -1] = input_numpy_array[0, :, 0]
    input_numpy_array[1, -1, :] = input_numpy_array[1, 0, :]
    input_numpy_array[1, :, -1] = input_numpy_array[1, :, 0]

  return input_numpy_array

def truncateFloat(f, n=7):
    if f >= 0:
      return np.floor(f * 10 ** n) / 10 ** n
    else:
      return -np.floor(abs(f) * 10 ** n) / 10 ** n

def partitionFunc1D(a, temperature):    # canonical partition function: exp(-U/kbT) / sigma(exp(-U/kbT))
  a = np.array(a)
  return np.exp(-(np.cos(a) + np.cos(2*a) + np.cos(3*a))/temperature) 

def partitionFunc2D(a, b, temperature): # canonical partition function: exp(-U/kbT) / sigma(exp(-U/kbT))
  a = np.array(a)
  b = np.array(b)
  x, y = symbols("x y") 
  Q = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2)) / temperature))  
  Q = lambdify([x, y], Q, "numpy")
  return Q(a, b) 

def boltz1D(a, temperature): # return probability
  a = np.array(a)
  q  = partitionFunc1D(a, temperature)
  q  = np.sum(q, axis=0)
  print(q)
  return np.exp(-(np.cos(a) + np.cos(2*a) + np.cos(3*a))/temperature)/q

def freeE1D(a, temperature):
  a = np.array(a)
  p = boltz1D(a, temperature) 
  return -1*temperature*np.log(p)

def boltz2D(a, b, temperature): # exp(-(K+U)/kbT) ~= exp(-K/kbT)exp(-U/kbT) ~= exp(-2/2) * exp(-U/kbT)
  a = np.array(a)
  b = np.array(b)
  q  = partitionFunc2D(a, b, temperature)
  q  = np.sum(q, axis=1)
  q  = np.sum(q, axis=0)
  print(q)
  x, y = symbols("x y") 
  fb = sympify(exp(-((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2)) / temperature))  
  fb = lambdify([x, y], fb, "numpy")
  return fb(a, b) / q

def freeE2D(a, b, temperature):
  a = np.array(a)
  b = np.array(b)
  p = boltz2D(a, b, temperature) 
  #res = -1*temperature*np.log(p)
  #res = paddingRighMostBins(res)
  return -1*temperature*np.log(p)
  #return res 
  
def Usurface1D(a):
  a = np.array(a)
  return np.cos(a) + np.cos(2*a) + np.cos(3*a) 
  
def Usurface2D(a, b):
  a = np.array(a)
  b = np.array(b)
  x, y = symbols("x y") 
  fU = sympify((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2))  
  fU = lambdify([x, y], fU, "numpy")
  return fU(a, b) 

def forcex1D(a):
  a = np.array(a)
  return np.sin(a) + 2*np.sin(2*a) + 3*np.sin(3*a) 

def forcex2D(a, b):
  a = np.array(a)
  b = np.array(b)
  x, y = symbols("x y") 
  fx = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), x)) 
  fx = lambdify([x,y], fx, "numpy")
  return -fx(a, b) 

def forcey2D(a, b):
  a = np.array(a)
  b = np.array(b)
  x, y = symbols("x y") 
  fy = sympify(diff((0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2), y)) 
  fy = lambdify([x,y], fy, "numpy")
  return -fy(a, b) 

if __name__ == "__main__":
  pass

 # bins = np.linspace(-np.pi, np.pi, 361)
  
 # T = 4 
 # freeE = freeE1D(bins, T)
 # with open("FreeE_1D_T%f.dat" %(T), "w") as fout:
 #   for b, f in zip(bins, freeE):
 #     fout.write(str(b) + " " + str(f) + "\n")

  
  import matplotlib.pyplot as plt

  binx = np.linspace(-2, 2, 201)
  biny = np.linspace(-2, 2, 201)
  #binx = np.linspace(-2, 2, 41)
  #biny = np.linspace(-2, 2, 41)
  binX, binY = np.meshgrid(binx, biny , indexing="ij")

  fx = forcex2D(binX, binY)
  fy = forcey2D(binX, binY)

  with open("estimate2D", "w") as fileOutProperty:
    for i in range(len(binx)):
      for j in range(len(biny)):
        fileOutProperty.write(str(binx[i]) + " ")
        fileOutProperty.write(str(biny[j]) + " ")
        fileOutProperty.write(str(fx[i][j]) + " " + str(fy[i][j]) + "\n")  

  print(fx)

  cs = plt.contourf(binX, binY, forcex2D(binX, binY), 8, cmap=plt.cm.plasma)
  R  = plt.contour(binX, binY, forcex2D(binX, binY), 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
  plt.show()
 
  
 
"""
  import matplotlib.pyplot as plt

  binx = np.linspace(-2, 2, 41)
  biny = np.linspace(-2, 2, 41)
  binX, binY = np.meshgrid(binx, biny , indexing="ij")
  
  temperature = 4
  U = freeE2D(binX, binY, temperature)


  with open("FreeE_2D_T4", "w") as fileOutProperty:
    for i in range(len(binx)):
      for j in range(len(biny)):
        fileOutProperty.write(str(binx[i]) + " ")
        fileOutProperty.write(str(biny[j]) + " ")
        fileOutProperty.write(str(U[i][j]) + "\n")  


  cs = plt.contourf(binX, binY, freeE2D(binX, binY, temperature), 8, cmap=plt.cm.plasma)
  R  = plt.contour(binX, binY, freeE2D(binX, binY, temperature), 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
  plt.show()
  """


  #boltz = boltz1D(bins, 0.05)
  #with open("boltz_1D_T0.05.dat", "w") as fout:
  # for b, f in zip(bins, boltz):
  #   fout.write(str(b) + " " + str(f) + "\n")
  #acc = 0
  #sq = 0
  #n = 100000
  #for i in range(n):
  # s = randMars() -0.5
  # acc += s
  # sq += s**2
  #print(acc/n)
  #print(sq/n)

