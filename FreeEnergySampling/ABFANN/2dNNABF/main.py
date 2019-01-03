#!/usr/bin/env python3

from ABF import importanceSampling
import os 

Ndims                 = 2

mass                  = 0.01
temperature           = 10 
frictCoeff            = 0.1
#time_length           = 2500 

learning_rate         = 0.001
regularCoeff          = 150 
epoch                 = 1000
trainingFreq          = 10000


#half_boxboundary      = 3.141592653589793 
#binNum                = 360 

half_boxboundary      = 3 
binNum                = 30 

box                   = [2*half_boxboundary, 2*half_boxboundary]
Nparticles            = 1
init_time             = 0.
time_step             = 0.005
init_frame            = 0
mode                  = "LangevinEngine"
NNoutputFreq          = 10


#abf_switch  = ["yes","no","yes"]
#NN_switch   = ["no","no","yes"]
#abf_switch  = ["yes","no"]
#NN_switch   = ["no","no"]
#abf_switch  = ["no"]
#NN_switch   = ["no"]
abf_switch  = ["yes"]
NN_switch   = ["no"]

force_distr = ["estimate"]
tl  = [50000]


############### Run Simulation & Validation Test ###############

for time_length in tl:
	for abfCheckFlag, nnCheckFlag in zip(abf_switch, NN_switch):

		filename_conventional = "conventional_" + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"
		filename_force        = "Force_"        + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat" 
		filename_Histogram    = "Histogram_"    + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"

		force_distr.append(filename_force)

		s = importanceSampling(Nparticles, Ndims, init_time, time_step, time_length, init_frame, mass, box, temperature, frictCoeff, abfCheckFlag, \
													nnCheckFlag, trainingFreq, mode, learning_rate, regularCoeff, epoch, NNoutputFreq, half_boxboundary, binNum, filename_conventional, filename_force) 
		s.mdrun()

		os.system("./checkBoltzmannDistr.py" + " " + str(Ndims) + " " + str(half_boxboundary) + " "  + str(binNum) + " " +  filename_conventional + " " + filename_Histogram)


	#os.system("./error.py"           + " " + force_distr[0]        + " "       + force_distr[1]    + " "         + force_distr[2] + " " + force_distr[3])
	os.system("./getTargetTemp.py"   + " " + str(mass)    + " "    + str(Ndims))
	os.system("mkdir"                + " " + str(Ndims)   + "D_"   + "m"       + str(mass)  + "T"                + str(temperature))
	os.system("./checkForceDistr.py" + " " + str(Ndims)   + " "    + str(half_boxboundary)  + " "  + str(binNum) + " " +  filename_force) 
	os.system("mv"                   + " " + "*.dat"               + " "       + str(Ndims) + "D_" + "m"         + str(mass) + "T"                 + str(temperature))
	os.system("mv"                   + " " + "*.png"               + " "       + str(Ndims) + "D_" + "m"         + str(mass) + "T"                 + str(temperature))

############### Run Simulation & Validation Test ###############






