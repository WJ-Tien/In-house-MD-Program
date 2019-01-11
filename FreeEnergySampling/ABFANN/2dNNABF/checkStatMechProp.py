#!/usr/bin/env python3

from render import rendering
from customMathFunc import getIndices, forcex2D, forcey2D, truncateFloat
import numpy as np
import os

class checkStatMechProp(object):
	
	def __init__(self, ndims, mass, half_boxboundary, binNum, abfCheckFlag, nnCheckFlag, temperature=None):
		self.temperature      = temperature
		self.ndims            = ndims
		self.half_boxboundary = half_boxboundary
		self.mass             = mass
		self.binNum           = binNum
		self.abfCheckFlag     = abfCheckFlag
		self.nnCheckFlag      = nnCheckFlag
		self.x_axis           = np.linspace(-half_boxboundary, half_boxboundary, binNum)
		self.y_axis           = np.linspace(-half_boxboundary, half_boxboundary, binNum)

	def checkBoltzDistr(self, fileIn, fileOut):

		with open(fileIn, "r") as fin:

			if self.ndims == 1: # use xmgrace instead
				prob_x = np.zeros((self.binNum), dtype = np.int32)  
				nsamples = 0

				for line in fin:
					line = line.split()
					if line[0] != "#":	
						nsamples += 1	
						prob_x[getIndices(truncateFloat(float(line[2])), self.x_axis)] += 1 

				prob_x = np.array(prob_x)
				prob_x = (prob_x / nsamples) # final probability distribution 

				with open(fileOut, "w") as fout:
					for i in range(len(prob_x)): # in xmgrace, one can discard the point on pi (boundary error)
						fout.write(str(self.x_axis[i]) + " " + str(prob_x[i]) + "\n")

			if self.ndims == 2:

				prob_xy  = np.zeros((self.binNum, self.binNum), dtype = np.float32)  
				nsamples = 0

				for line in fin:
					line = line.split()
					if line[0] != "#":	
						prob_xy[getIndices(truncateFloat(float(line[2])), self.x_axis)][getIndices(truncateFloat(float(line[4])), self.y_axis)] += 1 # 2 for cartcoord_1D, 2 4 for cartcoord_2D

				prob_xy = (prob_xy / nsamples / nsamples) # final probability distribution 

				with open(fileOut, "w") as fout:
					for i in range(self.binNum): # discard the point on the positive boundary (PBC issues)
						for j in range(self.binNum): # discard the point on the positive boundary (PBC issues)
							fout.write(str(self.x_axis[i]) + " ")
							fout.write(str(self.y_axis[j]) + " " +  str(prob_xy[i][j]) + "\n")

				prob_xy = np.delete(prob_xy, -1, 0) # rendering; prevent boundary error
				prob_xy = np.delete(prob_xy, -1, 1)

				r = rendering(self.ndims, self.half_boxboundary, self.binNum, self.temperature)
				r.render(prob_xy, name=str(self.abfCheckFlag + "_" + self.nnCheckFlag + "_" + "boltz2D"))			

	def checkForceDistr(self, fileIn):	

		with open(fileIn, "r") as fin:
			if self.ndims == 1:
				# use xmgrace instead
				pass	

			if self.ndims == 2:	
				force_x  = np.zeros((self.binNum, self.binNum), dtype=np.float32) 
				force_y  = np.zeros((self.binNum, self.binNum), dtype=np.float32) 
				i        = 0
				j        = 0

				for line in fin:
					line = line.split()
					force_x[i][j] = round(float(line[2]), 3)
					force_y[i][j] = round(float(line[4]), 3) 
					j += 1
					if j == self.binNum:
						j = 0
						i += 1
				
				force_x = np.delete(force_x, -1, 1) # rendering; prevent boundary error
				force_x = np.delete(force_x, -1, 0)
				force_y = np.delete(force_y, -1, 1)
				force_y = np.delete(force_y, -1, 0)
					
				r = rendering(self.ndims, self.half_boxboundary, self.binNum)
				r.render(force_x, name=str(self.abfCheckFlag + "_" + self.nnCheckFlag + "_" + "forcex2D"))			
				r.render(force_y, name=str(self.abfCheckFlag + "_" + self.nnCheckFlag + "_" + "forcey2D"))			

	def getTargetTemp(self, fileOut):
		curdir = os.getcwd()
		for filename in os.listdir(curdir):
			avg_vel_sq = 0.0 
			count      = 0
			if filename.startswith("conventional"):
				with open(filename, "r") as fin:
					if self.ndims == 1:
						for line in fin:
							line = line.split()
							if line[0] != "#":
								count += 1
								avg_vel_sq += ((float(line[3]))**2)

						avg_vel_sq /= (count)
						TargetTemp = (avg_vel_sq * (self.mass / self.ndims))

						with open(fileOut, "a") as fout:
							fout.write(str(filename) + "    " + str(TargetTemp) + "\n")

					if self.ndims == 2:
						for line in fin:
							line = line.split()
							if line[0] != "#":
								count += 1
								avg_vel_sq += ((float(line[3]))**2)
								avg_vel_sq += ((float(line[5]))**2)

						avg_vel_sq /= (count)
						TargetTemp = (avg_vel_sq * (self.mass / self.ndims))

						with open(fileOut, "a") as fout:
							fout.write(str(filename) + "    " + str(TargetTemp) + "\n")

	def relativeError(self, ideal_estimate, file_RegularMD, file_ABF, file_ABFANN, name):
		def readData(f):
			with open(f, "r") as fin:
				saveForce = []
				for line in fin:
					saveForce.append(float(line.split()[1]))
			return np.array(saveForce)

		id_est    = readData(ideal_estimate) # estimate
		ABFANN    = readData(file_ABFANN)    # Force_data: ABF with ANN
		ABF       = readData(file_ABF)       # Force_data: ABF 
		RegularMD = readData(file_RegularMD) # Force_data: No ABF No ANN 

		avg_ABFANN_error    = np.sqrt(np.sum(np.square(id_est - ABFANN))    / len(id_est))
		avg_ABF_error       = np.sqrt(np.sum(np.square(id_est - ABF))       / len(id_est))
		avg_RegularMD_error = np.sqrt(np.sum(np.square(id_est - RegularMD)) / len(id_est))

		with open(name, "w") as fout:
			fout.write("avg_ABFANN_error: "     + str(avg_ABFANN_error)    + "\n")
			fout.write("avg_ABF_error: "        + str(avg_ABF_error)       + "\n")
			fout.write("avg_noNN_noABF_error: " + str(avg_RegularMD_error) + "\n")

		
if __name__ == "__main__":
	pass
	#checkStatMechProp(ndims=2, mass=1, half_boxboundary=3, binNum=40, abfCheckFlag="no", nnCheckFlag="no").checkForceDistr("Force_m1_T0.1_noABF_noNN_TL_150000.dat")
	#checkStatMechProp(ndims=2, mass=1, half_boxboundary=3, binNum=40, abfCheckFlag="no", nnCheckFlag="no").checkBoltzDistr("conventional_m1_T0.1_noABF_noNN_TL_150000.dat", "Histogram_m1_T0.1_noABF_noNN_TL_150000.dat")
	#checkStatMechProp(ndims=2, mass=1, half_boxboundary=3, binNum=40, abfCheckFlag="yes", nnCheckFlag="no").checkForceDistr("Force_m1_T0.1_yesABF_noNN_TL_150000.dat")
	#checkStatMechProp(ndims=2, mass=1, half_boxboundary=3, binNum=40, abfCheckFlag="yes", nnCheckFlag="no").checkBoltzDistr("conventional_m1_T0.1_yesABF_noNN_TL_150000.dat", "Histogram_m1_T0.1_yesABF_noNN_TL_150000.dat")
	
