#!/usr/bin/env python3

from ABF import importanceSampling
import os 

mass                  = 1.0
temperature           = 5.0
time_length           = 2500 
frictCoeff            = 0.1

learning_rate         = 0.001
regularCoeff          = 150
epoch                 = 5000
trainingFreq          = 10000

Nparticles            = 1
Ndims                 = 1
init_time             = 0.
time_step             = 0.005
init_frame            = 0
box                   = [6.283185307179586]
mode                  = "LangevinEngine"
NNoutputFreq          = 10

abf_switch  = ["yes","yes","no"]
NN_switch   = ["yes","no","no"]
force_distr = ["estimate"]


############### Run Simulation & Validation Test ###############

for abfCheckFlag, nnCheckFlag in zip(abf_switch, NN_switch):

	filename_conventional = "conventional_" + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"
	filename_force        = "Force_"        + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat" 
	filename_Histogram    = "Histogram_"    + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"

	force_distr.append(filename_force)

	s = importanceSampling(Nparticles, Ndims, init_time, time_step, time_length, init_frame, mass, box, temperature, frictCoeff, abfCheckFlag, \
                        nnCheckFlag, trainingFreq, mode, learning_rate, regularCoeff, epoch, NNoutputFreq, filename_conventional, filename_force) 
	s.mdrun()

	os.system("./checkBoltzmannDistr.py" + " " +  filename_conventional + " " + filename_Histogram)

os.system("./error.py"         + " " + force_distr[0]        + " "       + force_distr[1] + " "       + force_distr[2] + " " + force_distr[3])
os.system("./getTargetTemp.py" + " " + str(mass))
os.system("mkdir"              + " " + "m"                   + str(mass) + "T"            + str(temperature))
os.system("mv"                 + " " + "*.dat"               + " "       + "m"            + str(mass) + "T"            + str(temperature))

############### Run Simulation & Validation Test ###############






