#!/usr/bin/env python3
import numpy as np
import time
from NN import trainingNN 

class importanceSampling(object):
	
	def __init__(self, particles, ndims, current_time, time_step, time_length, frame, mass, box, temperature, frictCoeff, abfCheckFlag, nnCheckFlag, trainingFrequency, mode, learning_rate, regularCoeff, epoch, NNoutputFreq, filename_conventional, filename_force):

		self.particles        = particles
		self.ndims            = ndims
		self.kb               = 1   # dimensionless kb
		self.binNum           = 360 # cut into 360 bins 
		self.binw             = 2 * np.pi / self.binNum
		self.colvars_coord    = np.array([np.arange(-np.pi, np.pi + self.binw, self.binw)] * self.ndims) # identical in xyz 
		self.colvars_force    = np.zeros((self.ndims, self.binNum+1), dtype=np.float32) # ixj
		self.colvars_force_NN = np.zeros((self.ndims, self.binNum+1), dtype=np.float32) 
		self.colvars_count    = np.zeros((self.ndims, self.binNum+1), dtype=np.int32) 
		self.current_coord    = np.zeros((self.particles, self.ndims), dtype=np.float32) 
		self.current_time     = current_time
		self.time_step        = time_step
		self.time_length      = time_length
		self.frame            = frame
		self.mass             = mass # database TODO 
		self.box              = np.array(box)  # list 
		self.temperature      = temperature
		self.pve_or_nve       = 1 if np.random.randint(1, 1001) % 2 == 0 else	-1
		self.current_vel      = np.array([[self.pve_or_nve * (np.sqrt(self.ndims * self.kb * self.temperature / self.mass)) for j in range(self.ndims)] for i in range(self.particles)], dtype=np.float32) 
		self.beta             = 1 / self.kb / self.temperature
		self.frictCoeff       = frictCoeff
		self.mode             = mode
		self.startTime        = time.time()
		self.abfCheckFlag     = abfCheckFlag 
		self.nnCheckFlag      = nnCheckFlag 
		self.Frequency        = trainingFrequency
		self.fileOut          = open(str(filename_conventional), "w") 
		self.fileOutForce     = open(str(filename_force), "w") 
		self.weightArr        = np.array([])
		self.biasArr          = np.array([])
		self.avg_vel_sq       = 0.
		self.learning_rate    = learning_rate 
		self.regularCoeff     = regularCoeff 
		self.epoch            = epoch 
		self.NNoutputFreq     = NNoutputFreq 

	def printIt(self):
		print("Frame %d with time %f" % (self.frame, time.time() - self.startTime)) 

	def writeFileHead(self):
		self.fileOut.write("#" + " " + "Mode"         + " " + str(self.mode)         + "\n")
		self.fileOut.write("#" + " " + "Dims"         + " " + str(self.ndims)        + "\n")
		self.fileOut.write("#" + " " + "Particles"    + " " + str(self.particles)    + "\n")
		self.fileOut.write("#" + " " + "temperature"  + " " + str(self.temperature)  + "\n") 
		self.fileOut.write("#" + " " + "frictCoeff"   + " " + str(self.frictCoeff)   + "\n") 
		self.fileOut.write("#" + " " + "Time_length"  + " " + str(self.time_length)  + "\n") 
		self.fileOut.write("#" + " " + "Time_step"    + " " + str(self.time_step)    + "\n") 
		self.fileOut.write("#" + " " + "abfCheckFlag" + " " + str(self.abfCheckFlag) + "\n")
		self.fileOut.write("#" + " " + "nnCheckFlag"  + " " + str(self.nnCheckFlag)  + "\n")
		self.fileOut.write("#" + " " + "Frame" + " "  + "Time" + " " + "Coordinates(xyz)" + " " + "Velocity(xyz)" + "\n")

	def conventionalDataOutput(self):
		for n in range(self.particles):	
			self.fileOut.write(str(self.frame) + " " + str(self.current_time) + " ")
			for d in range(self.ndims):
				self.fileOut.write(str(self.current_coord[n][d]) + " " + str(self.current_vel[n][d]) + "\n") 

	def forceOnColvarsOutput(self): #TODO
		
		self.colvars_force = (self.colvars_force / self.colvars_count)
		self.colvars_force[np.isnan(self.colvars_force)] = 0

		for d in range(self.ndims):
			for i in range(self.binNum):
				self.fileOutForce.write(str(self.colvars_coord[d][i]) + " ")
				self.fileOutForce.write(str(self.colvars_force[d][i]) + " " + str(self.colvars_count[d][i]) + " ")  
				self.fileOutForce.write("\n")

	def targetTempCheck(self):
		print(self.mass * (self.avg_vel_sq / self.frame) / self.ndims / self.kb)

	def PotentialForce(self, coord):     

		# For 1D toy model
		# Potential surface of the system: cosx + cos2x + cos3x
		# np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)
		# Reaction Coordinate (colvars) == Cartesian Coordinate, so Jacobian J = 1

		return np.sin(coord) + 2*np.sin(2*coord) + 3*np.sin(3*coord)

	def visForce(self, vel):
		return -self.frictCoeff * vel * self.mass

	def randForce(self): 
		random_constant = np.random.normal(0, 1)
		return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature / self.time_step) * (random_constant) 

	def calForce(self, coord, vel, d=None):

		if self.abfCheckFlag == "no" and self.nnCheckFlag == "yes": # (O)
			print("Something went wrong")

	
		# conventional LD (O)
		elif self.abfCheckFlag == "no" and self.nnCheckFlag == "no":  
			Fu = self.PotentialForce(coord) 
			Fsys = self.PotentialForce(coord) + self.visForce(vel) + self.randForce()
			self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += Fsys 
			self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1
			return Fu / self.mass 

	  # conventinal LD with ABF (O)
		elif self.abfCheckFlag == "yes" and self.nnCheckFlag == "no":
			Fu   = self.PotentialForce(coord) 
			Fsys = self.PotentialForce(coord) + self.visForce(vel) + self.randForce()
			self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += Fsys 
			self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1
			Fabf = -self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] / (self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2])
			return (Fu + Fabf) / self.mass

		# LD with ABF and NN (I)
		elif self.abfCheckFlag == "yes" and self.nnCheckFlag == "yes": 
			Fsys = self.PotentialForce(coord) + self.visForce(vel) + self.randForce()
			Fu   = self.PotentialForce(coord) 

			self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += Fsys 
			self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1

			if self.frame % self.Frequency == 0 and self.frame != 0: 
				output = trainingNN("loss.dat", "hyperparam.dat", "weight.pkl", "bias.pkl", self.ndims) 

				self.colvars_force = (self.colvars_force / self.colvars_count)
				self.colvars_force[np.isnan(self.colvars_force)] = 0 # 0/0 = nan n/0 = inf

				#output.training(self.weightArr, self.biasArr, self.colvars_coord, self.colvars_force, 0.001, 150, 4000, 10) 
				trainedWeightArr, trainedBiasArr, self.colvars_force_NN = \
				output.training(self.weightArr, self.biasArr, self.colvars_coord, self.colvars_force, self.learning_rate, self.regularCoeff, self.epoch, self.NNoutputFreq) 

				self.colvars_force  = (self.colvars_force * self.colvars_count)

				if self.weightArr.size == 0 and self.biasArr.size == 0:
					self.weightArr       = np.zeros((5, self.binNum+1), dtype=np.float32)
					self.biasArr         = np.zeros((5, self.binNum+1), dtype=np.float32)
					for i in range(5):
						self.weightArr[i] = trainedWeightArr[i]
						self.biasArr[i] = trainedBiasArr[i] 

				else:
					for i in range(5):
						self.weightArr[i] = trainedWeightArr[i]
						self.biasArr[i] = trainedBiasArr[i] 

				self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += (self.colvars_force_NN[d][int(np.floor(coord/self.binw)) + self.binNum//2]) 
				self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1
				Fabf = -self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] / self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] 
				return (Fu + Fabf) / self.mass

			else:
				Fabf = -(self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2]) / self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2]
				return (Fu + Fabf) / self.mass

		else: # typo handling
			print("something went wrong")
			exit(1)

	def LangevinEngine(self):

		# molecular_dynamics_2015.pdf (2018/12/16)
		# http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
		# https://pdfs.semanticscholar.org/f393/85336df44c2af1fd6f293540b18a701b1c56.pdf

		random_xi       = np.random.normal(0, 1)
		random_theta    = np.random.normal(0, 1)

		for n in range(self.particles):
			for d in range(self.ndims):

				self.avg_vel_sq            += ((self.current_vel[n][d])**2) # for targetTemp validation
				sigma                       = np.sqrt(2 * self.kb * self.temperature * self.frictCoeff / self.mass)
				current_force               = self.calForce(self.current_coord[n][d], self.current_vel[n][d], d) 
				Ct                          = (0.5*self.time_step**2) * (current_force - self.frictCoeff * self.current_vel[n][d]) + \
																			sigma * (self.time_step**1.5) * (0.5 * random_xi + (np.sqrt(3)/6) * random_theta)

				self.current_coord[n][d]    = self.current_coord[n][d] + (self.time_step * self.current_vel[n][d]) + Ct
				self.current_coord[n][d]   -= (round(self.current_coord[n][d] / self.box[d]) * self.box[d]) # PBC
				updated_force               = self.calForce(self.current_coord[n][d], self.current_vel[n][d], d) 
				self.current_vel[n][d]      = self.current_vel[n][d] + (0.5 * self.time_step * (current_force + updated_force)) - (self.time_step * self.frictCoeff * self.current_vel[n][d]) + \
                                      (np.sqrt(self.time_step) * sigma * random_xi)  - (self.frictCoeff * Ct)

		self.current_time += self.time_step 
		self.frame        += 1

	def mdrun(self):

		# the first frame
		self.writeFileHead()	
		self.conventionalDataOutput()
		self.printIt()

		# the rest of the frames
		while self.current_time < self.time_length:
			if self.mode == "LangevinEngine":
				self.LangevinEngine()
				self.conventionalDataOutput()		
				self.printIt()

		self.forceOnColvarsOutput()
		self.targetTempCheck()
		self.fileOut.close()
		self.fileOutForce.close()

if __name__ == "__main__":
	pass	
	# Particles, Ndim, current_time, time_step, time_length, fm, mass, box, temperature, frictCoeff, abfCheckFlag, nnCheckFlag, Frequency, mode, fname_conventional, fname_force):
	#import sys
	#s = importanceSampling(1, 1, 0., 0.005, float(sys.argv[3]), 0, 2.0, [6.283185307179586], 6., float(sys.argv[6]), sys.argv[4], sys.argv[5], 10000, "LangevinEngine", sys.argv[1], sys.argv[2]).mdrun()

