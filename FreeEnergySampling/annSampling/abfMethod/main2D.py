#!/usr/bin/env python3

from advSampling import ABF 
from checkStatMechProp import checkStatMechProp 
import os 

Ndims                 = 2
mass                  = 1 
temperature           = 5 

frictCoeff            = 0.001 

learningRate          = 0.031
epoch                 = 5000 
regularCoeff          = 0.00020 

switchSteps           = 20

lateLearningRate      = 0.031
lateEpoch             = 8000
lateRegularCoeff      = 0.0

trainingFreq          = 1000 

half_boxboundary      = 2.0 
binNum                = 41 
box                   = [2*half_boxboundary, 2*half_boxboundary]

Nparticles            = 1
init_time             = 0.
time_step             = 0.005
init_frame            = 0
mode                  = "LangevinEngine"
NNoutputFreq          = 100 
force_distr           = ["estimate"]
tl                    = [100000]

#abf_switch  = ["yes"]
#NN_switch   = ["yes"]
#abf_switch  = ["yes", "yes", "no"]
#NN_switch   = ["yes", "no", "no"]
abf_switch  = ["no"]
NN_switch   = ["no"]


for time_length in tl:
	for abfCheckFlag, nnCheckFlag in zip(abf_switch, NN_switch):

		filename_conventional = "conventional_" + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"
		filename_force        = "Force_"        + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat" 
		filename_Histogram    = "Histogram_"    + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"

		force_distr.append(filename_force)

		ABF(Nparticles, Ndims, init_time, time_step, time_length, init_frame, mass, box, temperature, frictCoeff, abfCheckFlag,\
			 nnCheckFlag, trainingFreq, mode, learningRate, regularCoeff, epoch, lateLearningRate, lateRegularCoeff, lateEpoch,\
       switchSteps, NNoutputFreq, half_boxboundary, binNum, filename_conventional, filename_force).mdrun()

		checkStatMechProp(Ndims, mass, half_boxboundary, binNum, abfCheckFlag, nnCheckFlag).checkForceDistr(filename_force)
		checkStatMechProp(Ndims, mass, half_boxboundary, binNum, abfCheckFlag, nnCheckFlag, temperature).checkBoltzDistr(filename_conventional, filename_Histogram)
		
	#if Ndims == 1:
		#checkStatMechProp(Ndims, mass, half_boxboundary, binNum, abfCheckFlag=None, nnCheckFlag=None).relativeError(force_distr[0], force_distr[1], force_distr[2], force_distr[3], "relativeError.dat")	

	checkStatMechProp(Ndims, mass, half_boxboundary, binNum, abfCheckFlag=None, nnCheckFlag=None).getTargetTemp("checkTargetTemp.dat")	
		
	os.system("mkdir" + " " + str(Ndims) + "D_" + "m"        + str(mass)  + "T"       + str(temperature))
	os.system("mv"    + " " + "*.dat"    + " "  + str(Ndims) + "D_" + "m" + str(mass) + "T" + str(temperature))
	os.system("mv"    + " " + "*.png"    + " "  + str(Ndims) + "D_" + "m" + str(mass) + "T" + str(temperature))



