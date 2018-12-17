#!/usr/bin/env python3

from ABF import importanceSampling
import os 


mass                  = 1.0
temperature           = 3.5
time_length           = 2500 
frictCoeff            = 0.1

learning_rate         = 0.001
regularCoeff          = 150
epoch                 = 4000

Nparticles            = 1
Ndims                 = 1
init_time             = 0.
time_step             = 0.005
init_frame            = 0
box                   = [6.283185307179586]
abfCheckFlag          = "yes"
nnCheckFlag           = "yes"
trainingFreq          = 10000
mode                  = "LangevinEngine"
NNoutputFreq          = 10
filename_conventional = "conventional_" + "m" + str(mass) + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + "_.dat"
filename_force        = "Force_"        + "m" + str(mass) + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + "_.dat" 
filename_Histogram    = "Histogram_"    + "m" + str(mass) + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + "_.dat"


### RUN ###

s = importanceSampling(Nparticles, Ndims, init_time, time_step, time_length, init_frame, mass, box, temperature, frictCoeff,\
											abfCheckFlag, nnCheckFlag, trainingFreq, mode, learning_rate, regularCoeff, epoch, NNoutputFreq, \
                      filename_conventional,filename_force) 

s.mdrun()

os.system("./checkBoltzmannDist.py" + " " +  filename_conventional + " " + "2" + " " + filename_Histogram)









