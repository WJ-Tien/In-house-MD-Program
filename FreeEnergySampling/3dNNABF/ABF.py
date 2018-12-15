#!/usr/bin/env python3
import numpy as np
import time
from NN import trainingNN 

class importanceSampling(object):
	
	def __init__(self, particles, ndims, current_coord, current_time, time_step, time_length, frame, mass, box, temperature, frictCoeff, abfCheckFlag, nnCheckFlag, Frequency, mode, filename_conventional, filename_force):

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
		self.box              = box  # list 
		self.temperature      = temperature
		self.pve_or_nve       = 1 if np.random.randint(1, 1001) % 2 == 0 else	-1
		self.current_vel      = np.array([[self.pve_or_nve * (np.sqrt(self.ndims * self.kb * self.temperature / self.mass)) for j in range(self.ndims)] for i in range(self.particles)], dtype=np.float32) 
		self.beta             = 1 / self.kb / self.temperature
		self.frictCoeff       = frictCoeff
		self.mode             = mode
		self.startTime        = time.time()
		self.abfCheckFlag     = abfCheckFlag 
		self.nnCheckFlag      = nnCheckFlag 
		self.Frequency        = Frequency
		self.fileOut          = open(str(filename_conventional), "w") 
		self.fileOutForce     = open(str(filename_force), "w") 
		self.weightList       = np.array([])
		self.biasList         = np.array([])

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
		self.fileOut.write("#" + " " + "Frame" + " "  + "Time" + " " + "Coordinates(xyz)" + " " + "Velocity(xyz)" + " " + "Fluctuated Temperature" + "\n")

	def conventionalDataOutput(self):
		avg_vel_sq = 0 
		for n in range(self.particles):	
			self.fileOut.write(str(self.frame) + " " + str(self.current_time) + " ")
			for d in range(self.ndims):
				self.fileOut.write(str(self.current_coord[n][d]) + " " + str(self.current_vel[n][d]) + " ")
				avg_vel_sq += self.current_vel[n][d]**2
		avg_vel_sq /= self.particles
		self.fileOut.write(str((1/self.ndims) *self.mass*avg_vel_sq / self.kb) + "\n")

	def forceOnColvarsOutput(self): #TODO
		self.colvars_force = (self.colvars_force / self.colvars_count)
		self.colvars_force[np.isnan(self.colvars_force)] = 0

		for d in range(self.ndims):
			for i in range(self.binNum):
				self.fileOutForce.write(str(self.colvars_coord[d][i]) + " ")
				self.fileOutForce.write(str(self.colvars_force[d][i]) + " " + str(self.colvars_count[d][i]) + " ")  
				self.fileOutForce.write("\n")

	def PotentialForce(self, coord):     

		# For 1D toy model
		# Potential surface of the system: cosx + cos2x + cos3x
		# np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)
		# Reaction Coordinate (colvars) == Cartesian Coordinate, so Jacobian J = 1

		return np.sin(coord) + 2*np.sin(2*coord) + 3*np.sin(3*coord)

	def visForce(self, vel):
		return -self.frictCoeff * vel * self.mass

	def randForce(self): 
		random_xi    = np.random.normal(0, 1)
		random_theta = np.random.normal(0, 1)
		return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature) * (random_theta) 

	def calForce(self, coord, vel, d=None):

		if self.abfCheckFlag == "no" and self.nnCheckFlag == "yes": # (O)
			print("Something went wrong")
			exit(1)

	
		# conventional LD (O)
		elif self.abfCheckFlag == "no" and self.nnCheckFlag == "no":  
			Fu = self.PotentialForce(coord) 
			Fsys = self.PotentialForce(coord) + self.visForce(vel) + self.randForce()
			self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += Fsys 
			self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1
			return Fu 

	  # conventinal LD with ABF (O)
		elif self.abfCheckFlag == "yes" and self.nnCheckFlag == "no":
			Fsys = self.PotentialForce(coord) + self.visForce(vel) + self.randForce()
			Fu   = self.PotentialForce(coord)
			self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += Fsys 
			self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1
			Fabf = -self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] / (self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2])
			return Fu + Fabf 

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

				trainedWeightArr, trainedBiasArr, self.colvars_force_NN = \
				output.training(self.weightList, self.biasList, self.colvars_coord, self.colvars_force, 0.0005, 10, 1000, 10) #TODO NN.py

				self.colvars_force  = (self.colvars_force * self.colvars_count)

				if self.weightList.size == 0 and self.biasList.size == 0:
					self.weightList       = np.zeros((5, self.binNum+1), dtype=np.float32)
					self.biasList         = np.zeros((5, self.binNum+1), dtype=np.float32)
					for i in range(5):
						self.weightList[i] = trainedWeightArr[i]
						self.biasList[i] = trainedBiasArr[i] 

				else:
					for i in range(5):
						self.weightList[i] = trainedWeightArr[i]
						self.biasList[i] = trainedBiasArr[i] 

				self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] += (self.colvars_force_NN[d][int(np.floor(coord/self.binw)) + self.binNum//2]) 
				self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] += 1
				Fabf = -self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2] / self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2] 
				return Fu + Fabf 

			else:
				Fabf = -(self.colvars_force[d][int(np.floor(coord / self.binw)) + self.binNum//2]) / self.colvars_count[d][int(np.floor(coord / self.binw)) + self.binNum//2]
				return Fu + Fabf 

		else: # typo handling
			print("something went wrong")
			exit(1)

	def LangevinEngine(self):

		# http://www.complexfluids.ethz.ch/Langevin_Velocity_Verlet.pdf (2018/11/22)

		a                  = (2 - self.frictCoeff * self.time_step) / (2 + self.frictCoeff * self.time_step)
		b                  = np.sqrt(self.kb * self.temperature * self.frictCoeff * self.time_step / 2)
		c                  = (2 * self.time_step) / (2 + (self.frictCoeff * self.time_step))
		random_eta         = np.random.normal(0, 1)

		for n in range(self.particles):
			for d in range(self.ndims):
				current_force               = self.calForce(self.current_coord[n][d], self.current_vel[n][d], d)
				self.current_vel[n][d]      = self.current_vel[n][d] + (current_force * self.time_step / 2) + (b * random_eta)    # half time step
				self.current_coord[n][d]    = self.current_coord[n][d] + c * self.current_vel[n][d] # full time step 
				self.current_coord[n][d]   -= (round(self.current_coord[n][d] / self.box[d]) * self.box[d]) # PBC
				current_force               = self.calForce(self.current_coord[n][d], self.current_vel[n][d], d)
				self.current_vel[n][d]      = (a *self.current_vel[n][d]) + (b * random_eta) + (self.time_step / 2) * current_force # full time_step

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

		self.fileOut.close()
		self.fileOutForce.close()

if __name__ == "__main__":
	
	# Particles, Ndim, current_coord current_time, time_step, time_length, fm, mass, box, temperature, frictCoeff, abfCheckFlag, nnCheckFlag, Frequency, mode, fname_conventional, fname_force):
	import sys
	s = importanceSampling(1, 1, [0.], 0., 0.005, float(sys.argv[3]), 0, 2., [6.283185307179586], 1., float(sys.argv[6]), sys.argv[4], sys.argv[5], 250, "LangevinEngine", sys.argv[1], sys.argv[2]).mdrun()


