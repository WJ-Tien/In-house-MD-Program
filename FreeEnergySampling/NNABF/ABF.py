#!/usr/bin/env python3
import numpy as np
import time

class importanceSampling(object):
	
	def __init__(self, current_coord, current_time, time_step, time_length, frame, mass, boxsize_x, temperature, frictCoeff, abfCheckFlag, mode, filename_conventional, filename_force):

		self.fileOut         = open(str(filename_conventional), "w") 
		self.fileOutForce    = open(str(filename_force), "w") 
		self.kb              = 1   # dimensionless kb
		self.binNum          = 360 # cut into 360 bins 
		self.binw            = 2 * np.pi / self.binNum
		self.force_storage   = [0] * (self.binNum + 1)
		self.colvars_count   = [0] * (self.binNum + 1)
		self.colvars_coord   = np.arange(-np.pi, np.pi + 2*np.pi/self.binNum, 2*np.pi/self.binNum)
		self.current_coord   = current_coord
		self.current_time    = current_time
		self.time_step       = time_step
		self.time_length     = time_length
		self.frame           = frame
		self.mass            = mass
		self.boxsize_x       = boxsize_x
		self.temperature     = temperature
		self.current_vel     = np.sqrt(self.kb * self.temperature / self.mass)
		self.frictCoeff      = frictCoeff
		self.abfCheckFlag    = abfCheckFlag 
		self.mode            = mode
		self.startTime       = time.time()

	def printIt(self):
		print("Frame %d with time %f" % (self.frame, time.time() - self.startTime))

	def writeFileHead(self):
		self.fileOut.write("#" + " " + "Mode"         + " " + str(self.mode)         + "\n")
		self.fileOut.write("#" + " " + "temperature"  + " " + str(self.temperature)  + "\n") 
		self.fileOut.write("#" + " " + "gamma"        + " " + str(self.frictCoeff)   + "\n") 
		self.fileOut.write("#" + " " + "Time_length"  + " " + str(self.time_length)  + "\n") 
		self.fileOut.write("#" + " " + "Time_step"    + " " + str(self.time_step)    + "\n") 
		self.fileOut.write("#" + " " + "abfCheckFlag" + " " + str(self.abfCheckFlag) + "\n")
		self.fileOut.write("#" + " " + "Frame" + " "  + "Time" + " " + "Coordinates" + " " + "Velocity" + " " + "Fluctuated Temperature" + "\n")

	def conventionalDataOutput(self):
		self.fileOut.write(str(self.frame) + " " + str(self.current_time) + " " + str(self.current_coord) + " " + str(self.current_vel) + " " + str(self.mass*self.current_vel**2 / self.kb) + "\n")

	def forceOnColvarsOutput(self):
		for i in range(self.binNum):
			try:
				self.fileOutForce.write(str(self.colvars_coord[i]) + " " + str(self.colvars_count[i]) + " " + str(self.force_storage[i]) + " " + str(self.force_storage[i] / self.colvars_count[i]) + "\n") 
			except:
				self.fileOutForce.write(str(self.colvars_coord[i]) + " " + str(self.colvars_count[i]) + " " + str(self.force_storage[i]) + " " + str("0") + "\n")

	def PotentialForce(self, CartesianCoord):     

		# For 1D toy model
		# Potential surface of the system: cosx + cos2x + cos3x
		# np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)
		# Reaction Coordinate (colvars) == Cartesian Coordinate  so Jacobian J = 1
		return np.sin(CartesianCoord) + 2*np.sin(2*CartesianCoord) + 3*np.sin(3*CartesianCoord)

	def visForce(self):
		return -self.frictCoeff * self.current_vel * self.mass

	def randForce(self, random_xi, random_theta):
		return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature) * (0.5 * random_xi + 0.288675 * random_theta) 

	def LangevinEngine(self):

		random_theta = np.random.normal(0, 1)
		random_xi = np.random.normal(0, 1)

		if self.abfCheckFlag == "yes":
			self.force_storage[int(np.floor(self.current_coord / self.binw)) + self.binNum//2] += (self.PotentialForce(self.current_coord) + self.visForce() + self.randForce(random_xi, random_theta)) 
			self.colvars_count[int(np.floor(self.current_coord / self.binw)) + self.binNum//2] += 1
			current_force = 0. # biased force applied
		else:
			current_force = self.PotentialForce(self.current_coord)

		sigma = np.sqrt(2.0 * self.kb * self.temperature * self.frictCoeff / self.mass)

		Ct = (0.5 * self.time_step * self.time_step * (current_force / self.mass - self.frictCoeff * self.current_vel)) + \
         sigma * (self.time_step**1.5) * (0.5 * random_xi + 0.288675 * random_theta) 

		next_coord = self.current_coord + self.time_step * self.current_vel + Ct
		next_coord -= (round(next_coord / self.boxsize_x) * self.boxsize_x) # PBC

		if self.abfCheckFlag == "yes":
			next_force = 0. # biased force applied 
		else:
			next_force = self.PotentialForce(next_coord)

		next_vel = self.current_vel + (0.5 * self.time_step * (next_force + current_force) / self.mass) - \
							 self.time_step * self.frictCoeff * self.current_vel + sigma * np.sqrt(self.time_step) * random_xi - \
               self.frictCoeff * Ct 
			
		next_time = self.current_time + self.time_step 
		
		self.current_coord = next_coord
		self.current_vel = next_vel
		self.current_time = next_time
		self.frame += 1

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

	# current_coord, current_time, time_step, time_length, fm, mass, boxsize_x, temperature, frictCoeff, abfCheckFlag, mode, fname_conventional, fname_force):
	# boxsize_x ranges from -pi ~ pi

	s = importanceSampling(0., 0., 0.005, 50000, 0, 1., 6.283185307179586, 4, 1., "yes", "LangevinEngine", "wABF000.dat", "wABF_force000.dat").mdrun()

