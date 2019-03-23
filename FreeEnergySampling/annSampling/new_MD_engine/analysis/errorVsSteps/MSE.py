#!/usr/bin/env python3
from sys import argv
import numpy as np

if len(argv) < 4:
  print("typeCol <2 or 3 or 4> <ideal_estimate_file> <target_file>")
  print("typeCol 2: x fx")
  print("typeCol 3: x y fx")
  print("typeCol 4: x y fx fy")
  print("1D force: type 2")
  print("1D freeE: type 3")
  print("2D freeE: type 3")
  print("2D force: type 4")
  exit(1)

step         = []
typeCol      = int(argv[1])

# typecol = 2; 1D
if typeCol == 2:
  standard_1DX = []
  record_1DX   = []
  error_1DX    = []

# typeCol = 3; 2D
if typeCol == 3:
  standard_2DX = []
  record_2DX   = []
  error_2DX    = []

# typeCol = 4; 2D
if typeCol == 4:
  standard_2DX = []
  standard_2DY = []
  record_2DX   = []
  record_2DY   = []
  error_2DX    = []
  error_2DY    = []

# read standard (ideal estimate)
with open(argv[2], "r") as fin: 

  for line in fin:

    line = line.split()

    if typeCol == 2:
      standard_1DX.append(float(line[1])) # x fx

    if typeCol == 3: 
      standard_2DX.append(float(line[2])) # x y fx

    if typeCol == 4:
      standard_2DX.append(float(line[2])) # x y fx fy
      standard_2DY.append(float(line[3]))

  try:
    standard_1DX = np.array(standard_1DX)
    standard_2DX = np.array(standard_2DX)
    standard_2DY = np.array(standard_2DY)
  except NameError:
    pass

# read target(trained/untrained)
with open(argv[3], "r") as fin: 

  for line in fin:

    if line[0] == "#": # read steps
      step.append(line.split()[1])

    if len(line.strip()) == 0: # calculate MSE

      if typeCol == 2:
        record_1DX = np.array(record_1DX)
        record_1DX = (record_1DX - standard_1DX)**2

        MSE_1DX = np.sum(record_1DX) / record_1DX.shape[0] 
        error_1DX.append(MSE_1DX)

        record_1DX = list(record_1DX)
        record_1DX = []

      if typeCol == 3:
        record_2DX = np.array(record_2DX)
        record_2DX = (record_2DX - standard_2DX)**2

        MSE_2DX = np.sum(record_2DX) / record_2DX.shape[0] 
        error_2DX.append(MSE_2DX)

        record_2DX = list(record_2DX)
        record_2DX = []

      if typeCol == 4:
        record_2DX = np.array(record_2DX) 
        record_2DY = np.array(record_2DY) 
        record_2DX = (record_2DX - standard_2DX) ** 2
        record_2DY = (record_2DY - standard_2DY) ** 2

        MSE_2DX = np.sum(record_2DX) / record_2DX.shape[0] 
        MSE_2DY = np.sum(record_2DY) / record_2DY.shape[0] 
        error_2DX.append(MSE_2DX)
        error_2DY.append(MSE_2DY)

        record_2DX = list(record_2DX)
        record_2DY = list(record_2DY)
        record_2DX = []
        record_2DY = []

    if len(line.strip()) != 0 and line[0] != "#": # read colvars_data 

      line = line.split()

      if typeCol == 2:
        record_1DX.append(float(line[1]))

      if typeCol == 3:
        record_2DX.append(float(line[2]))

      if typeCol == 4:
        record_2DX.append(float(line[2]))
        record_2DY.append(float(line[4]))
        
# write the results
with open("ERROR_" + argv[3][0:len(argv[3]) - 4] + ".dat", "w") as fout:

  if typeCol == 2:
    for i, j in zip(step, error_1DX):
      fout.write(str(i) + " " + str(j) + "\n")

  if typeCol == 3:
    for i, j in zip(step, error_2DX):
      fout.write(str(i) + " " + str(j) + "\n")

  if typeCol == 4:      
    for i, j, k in zip(step, error_2DX, error_2DY):
      fout.write(str(i) + " " + str(j) + " " + str(k) + "\n")

