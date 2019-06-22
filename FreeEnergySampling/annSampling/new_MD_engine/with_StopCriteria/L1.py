#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def readFile(f, ndims, fflag=False, ideal=False):
  i = 0
  j = 0
  MAGIC = 41
  CV_1D = [] 
  CV_2D = np.zeros((2,MAGIC,MAGIC))

  if fflag == False:
    FES_1D = [] 
    FES_2D = np.zeros((MAGIC,MAGIC)) 
  else:
    Force_1D = []
    Force_2D = np.zeros((2,MAGIC,MAGIC))

  with open(f, "r") as fin:
    for line in fin:
      line = line.split()
      # FES
      if fflag == False: 

        if ndims == 1:
          CV_1D.append(float(line[0]))
          FES_1D.append(float(line[1]))

        if ndims == 2:
          CV_2D[0][i][j] = float(line[0])
          CV_2D[0][i][j] = float(line[1])
          FES_2D[i][j]   = float(line[2])
          j += 1
          if j == MAGIC:
            i += 1
            j = 0

      # Force
      else: 

        if ndims == 1:
          CV_1D.append(float(line[0]))
          Force_1D.append(float(line[1]))

        if ndims == 2:
          CV_2D[0][i][j]    = float(line[0])
          CV_2D[0][i][j]    = float(line[1])
          Force_2D[0][i][j] = float(line[2])
          if ideal:
            Force_2D[1][i][j] = float(line[3])
          else:
            Force_2D[1][i][j] = float(line[4])
          j += 1
          if j == MAGIC:
            i += 1
            j = 0

  if fflag == False and ndims == 1:
    return CV_1D, FES_1D

  if fflag == False and ndims == 2:
    return CV_2D, FES_2D

  if fflag and ndims == 1:
    return CV_1D, Force_1D

  if fflag and ndims == 2:
    return CV_2D, Force_2D

def writeFile(f, CV, target, ndims, fflag=False):
  MAGIC = 41

  # FES
  if fflag == False:
    if ndims == 1:
      with open(f, "w") as fout:
        for i, j in zip(CV, target):
          fout.write(str(i) + " " + str(j) + "\n")

    if ndims == 2:
      with open(f, "w") as fout:
        for i in range(MAGIC):
          for j in range(MAGIC):
            fout.write(str(CV[0][i][j]) + " " + str(CV[1][i][j]) + " " + str(target[i][j]) + "\n")

  # Force
  else:
    if ndims == 1:
      with open(f, "w") as fout:
        for i, j in zip(CV, target):
          fout.write(str(i) + " " + str(j) + "\n")

    if ndims == 2:
      with open(f, "w") as fout:
        for i in range(MAGIC):
          for j in range(MAGIC):
            fout.write(str(CV[0][i][j]) + " " +  str(CV[1][i][j]) + " " + str(target[0][i][j]) + " " + str(target[1][i][j]) + "\n")

def abs_L1_loss(ref, tg):
  ref = np.array(ref)
  tg  = np.array(tg)
  return abs(tg - ref)


def contourplt(fig, loc, CVX, CVY, input_arr, tname, pname):
  """ the subplots """
  # aspect=equal to ensure to render the square plot
  ax = fig.add_subplot(loc, aspect='equal') 

  ax.tick_params(labelsize=15)
  ax.set_title(tname, fontsize=20, y=1.04)
  cs = ax.contourf(CVX, CVY, input_arr, 8, cmap=plt.cm.plasma)
  R  = ax.contour(CVX, CVY, input_arr, 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
  ax.set_xlabel("X", fontsize=20, labelpad=15)
  ax.set_ylabel("Y", fontsize=20, labelpad=15)
  #ax.clabel(R, inline=True, fontsize=7.5)

  # align the colorbar to the subplot
  cb = plt.colorbar(cs, fraction=0.046, pad=0.04) 
  cb.ax.tick_params(labelsize=15)
  plt.savefig(pname)

if __name__ == "__main__":

  from sys import argv

  if len(argv) < 5:
    print("<f or fes> <ideal File> <target File> <ndims>")
    exit(1)

  idealFile  = argv[2]
  targetFile = argv[3]
  ndims      = int(argv[4])

  if argv[1] == "f":
    CV, ref = readFile(idealFile, ndims , True, True)  # ideal f
    _, tg   = readFile(targetFile, ndims , True) # target f
    loss = abs_L1_loss(ref, tg)
    writeFile("abs_L1_loss_force.dat", CV, loss, ndims, True)
    if ndims == 2:
      fig = plt.figure(figsize=(10, 8))
      tmpCVX = np.linspace(-2, 2, 41)
      tmpCVY = np.linspace(-2, 2, 41)
      CVX, CVY = np.meshgrid(tmpCVX, tmpCVY, indexing="ij")
      contourplt(fig, 111, CVX, CVY, loss[0], "abs L1 Loss of $\mathregular{f_x}$", "fx.png")
      plt.gcf().clear()
      contourplt(fig, 111, CVX, CVY, loss[1], "abs L1 Loss of $\mathregular{f_y}$", "fy.png")
       
  if argv[1] == "fes":
    CV, ref = readFile(idealFile, ndims , False)  # ideal fes
    _, tg   = readFile(targetFile, ndims , False) # target fes
    loss = abs_L1_loss(ref, tg)
    writeFile("abs_L1_loss_FES.dat", CV, loss, ndims, False)
    if ndims == 2:
      fig = plt.figure(figsize=(10, 8))
      tmpCVX = np.linspace(-2, 2, 41)
      tmpCVY = np.linspace(-2, 2, 41)
      CVX, CVY = np.meshgrid(tmpCVX, tmpCVY, indexing="ij")
      contourplt(fig, 111, CVX, CVY, loss, "abs L1 Loss of Free Energy", "FreeE.png")


