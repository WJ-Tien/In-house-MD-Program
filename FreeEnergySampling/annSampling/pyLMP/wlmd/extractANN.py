#!/usr/bin/env python3
import numpy as np

blankFlag = True 
step = []
annData = []
annStandard = []
errorANN = []

with open("annStandard.dat", "r") as fin:
  for line in fin:
    annStandard.append(float(line.split()[1]))

with open("annST.dat", "r") as fin:

  for line in fin:
    if len(line.strip()) == 0:
      blankFlag = True
      annData = np.array(annData)
      annStandard = np.array(annStandard)
      annData = (annData - annStandard) ** 2
      annMSE = np.sum(annData) / annData.shape[0]
      errorANN.append(annMSE)
      annData = list(annData)
      annData = []

    if line[0] == "#" and blankFlag == True: 
      step.append(float((line.split()[2])))
      blankFlag = False

    if line[0] != "#" and len(line.strip()) != 0:
      line = line.split()
      annData.append(float(line[1]))


with open("annOut.dat", "w") as fout:
  for i, j in zip(step, errorANN):
    fout.write(str(i) + " " + str(j) + "\n")
