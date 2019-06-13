#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

"""
# https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
# https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
p1 p2
p3 p4
"""
def read2D(f, flag=None):
  if flag == "fs":
    arr  = np.zeros((2,41,41))
  else:
    arr = np.zeros((41,41))

  with open(f, "r") as fin:
    i = 0
    j = 0
    for line in fin:
      line = line.split()

      if flag == "fs": 
        arr[0][i][j] = float(line[2])
        arr[1][i][j] = float(line[4])

      else:
        arr[i][j] = float(line[2])

      j += 1
      if j == 41:
        j = 0
        i += 1
  return arr

def mainplt(fig, arrangement): 
  """ main template """
  ax = fig.add_subplot(111)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

  if arrangement == "vertical": # TODO
    ax.set_xlabel("Cartesian coordinate X", fontsize=20, labelpad=20)

  if arrangement == "horizontal":
    ax.set_ylabel("Cartesian coordinate Y", fontsize=20, labelpad=20)

def contourplt(fig, loc, CVX, CVY, input_arr, tname, arrangement):
  """ the subplots """
  # aspect=equal to ensure to render the square plot
  ax = fig.add_subplot(loc, aspect='equal') 

  ax.tick_params(labelsize=12)
  ax.set_title(tname, fontsize=20, y=1.04)
  cs = ax.contourf(CVX, CVY, input_arr, 8, cmap=plt.cm.plasma)
  R  = ax.contour(CVX, CVY, input_arr, 8, colors='black', linewidth=.25, linestyles="solid", extend="both")
  ax.clabel(R, inline=True, fontsize=8.5)

  # align the colorbar to the subplot
  cb = plt.colorbar(cs, fraction=0.046, pad=0.04) 
  cb.ax.tick_params(labelsize=12)

  # space betwen subplots
  if arrangement == "vertical": # TODO 
    plt.subplots_adjust(hspace=0.30) 
    ax.set_ylabel("Cartesian coordinate Y", fontsize=20, labelpad=8)

  if arrangement == "horizontal":
    plt.subplots_adjust(wspace=0.30) 
    ax.set_xlabel("Cartesian coordinate X", fontsize=20, labelpad=8)

if __name__ == "__main__":
  from sys import argv

  # generate CV
  CVX      = np.linspace(-2, 2, 41)
  CVY      = np.linspace(-2, 2, 41)
  CVX, CVY = np.meshgrid(CVX, CVY, indexing="ij")

  # read input 
  noannFS  = read2D("Force_m0.1T4.000_gamma0.5000_len_5600000_yes_no.dat", "fs")
  annFS    = read2D("Force_m0.1T4.000_gamma0.5000_len_5600000_yes_yes.dat", "fs")

  # render contour plots
  if argv[1] == "h": 
    arrangement = "horizontal"
    fig = plt.figure(figsize=(12, 6))
    mainplt(fig, arrangement)
    contourplt(fig, 121, CVX, CVY, noannFS[0], "Force along X", arrangement)
    contourplt(fig, 122, CVX, CVY, noannFS[1], "Force along Y", arrangement)

  if argv[1] == "v": 
    arrangement = "vertical"
    fig = plt.figure(figsize=(6, 12))
    mainplt(fig, arrangement)
    contourplt(fig, 211, CVX, CVY, annFS[0], "Force along X", arrangement)
    contourplt(fig, 212, CVX, CVY, annFS[1], "Force along Y", arrangement)

  # save plots
  plt.savefig(arrangement + ".eps")
  plt.savefig(arrangement + ".png")

