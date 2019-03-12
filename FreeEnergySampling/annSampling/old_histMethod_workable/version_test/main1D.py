#!/usr/bin/env python3

from advSampling import ABP
from checkStatMechProp import checkStatMechProp 
import os 

Ndims                 = 1
mass                  = 1 
temperature           = 5 

frictCoeff            = 0.05 

learningRate          = 0.075 
epoch                 = 2500 
regularCoeff          = 0.00025 

switchSteps           = 10 
trainingFreq          = 25000 

lateLearningRate      = 0.075
lateEpoch             = 5000
lateRegularCoeff      = 0.0


half_boxboundary      = 3.141592653589793 
binNum                = 361 
box                   = [2*half_boxboundary]

Nparticles            = 1
init_time             = 0.
time_step             = 0.005
init_frame            = 0
mode                  = "LangevinEngine"
NNoutputFreq          = 100 
force_distr           = ["estimate"]
tl                    = [25000]

abf_switch  = ["no", "yes"]
NN_switch   = ["no", "yes"]
#abf_switch  = ["yes", "no"]
#NN_switch   = ["yes","no"]
#abf_switch  = ["no"]
#NN_switch   = ["no"]


for time_length in tl:
	for abfCheckFlag, nnCheckFlag in zip(abf_switch, NN_switch):

		filename_conventional = "conventional_" + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"
		filename_force        = "Force_"        + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat" 
		filename_Histogram    = "Histogram_"    + "m" + str(mass) + "_" + "T" + str(temperature) + "_" + abfCheckFlag + "ABF_" + nnCheckFlag + "NN_" + "TL_" + str(time_length) + ".dat"

		force_distr.append(filename_force)

		#ABF(Nparticles, Ndims, init_time, time_step, time_length, init_frame, mass, box, temperature, frictCoeff, abfCheckFlag,\
		#	 nnCheckFlag, trainingFreq, mode, learningRate, regularCoeff, epoch, lateLearningRate, lateRegularCoeff, lateEpoch,\
    #   switchSteps, NNoutputFreq, half_boxboundary, binNum, filename_conventional, filename_force).mdrun()
		ABP(Nparticles, Ndims, init_time, time_step, time_length, init_frame, mass, box, temperature, frictCoeff, abfCheckFlag,\
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



