#!/usr/bin/env python3
import numpy as np
import os

blankFlag = True 
step = []
ehistData = []
annStandard = []
errorSTMD = []

curdir = os.getcwd()
fileTank = []

with open("annStandard.dat") as fin:
  for line in fin:
    line = line.split()
    annStandard.append(float(line[1]))
  annStandard = np.array(annStandard)

for filename in os.listdir(curdir):
  if filename.startswith("Ehist") and not filename.endswith("restart"):
    fileTank.append(filename) 

fileTank.sort(key=lambda x: int(x[6:]))      

for i in fileTank:
  step.append(int(i[6:]))
  with open(i, "r") as fin:
    for line in fin:
      if line[0] != "#":
        line = line.split()
        ehistData.append(float(line[1]))

  ehistData = np.array(ehistData)
  ehistData = (ehistData - annStandard) ** 2
  MSE   = np.sum(ehistData) / ehistData.shape[0]
  errorSTMD.append(MSE)
  ehistData = list(ehistData)
  ehistData = []

with open("ehistOut.dat", "w") as fout:
  for i, j in zip(step, errorSTMD):
    fout.write(str(i) + " " + str(j) + "\n")

