#!/usr/bin/env python3
import numpy as np
import time

class importanceSampling(object):
	
	def __init__(self, current_coord, current_time, time_step, time_length, frame, mass, boxsize_x, temperature, frictCoeff, abfCheckFlag, mode, filename_conventional, filename_force):

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
		self.beta            = 1 / self.kb / self.temperature
		self.frictCoeff      = frictCoeff
		self.mode            = mode
		self.startTime       = time.time()
		self.abfCheckFlag    = abfCheckFlag 
		self.fileOut         = open(str(filename_conventional), "w") 
		self.fileOutForce    = open(str(filename_force), "w") 

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

	def PotentialForce(self, coord):     

		# For 1D toy model
		# Potential surface of the system: cosx + cos2x + cos3x
		# np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)
		# Reaction Coordinate (colvars) == Cartesian Coordinate  so Jacobian J = 1
		return np.sin(coord) + 2*np.sin(2*coord) + 3*np.sin(3*coord)

	def visForce(self, vel):
		return -self.frictCoeff * vel * self.mass

	def randForce(self):
		random_xi    = np.random.normal(0, 1)
		random_theta = np.random.normal(0, 1)
		return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature) * (0.333333 * random_xi + 0.288675 * random_theta) 

	def calForce(self, coord, vel):

		if self.abfCheckFlag == "yes":
			self.force_storage[int(np.floor(coord / self.binw)) + self.binNum//2] += (self.PotentialForce(coord) + self.visForce(vel) + self.randForce()) 
			self.colvars_count[int(np.floor(coord / self.binw)) + self.binNum//2] += 1
			return self.visForce(vel) + self.randForce()

		else:
				return self.PotentialForce(coord) + self.visForce(vel) + self.randForce() 

	def LangevinEngine(self):

		# http://www.complexfluids.ethz.ch/Langevin_Velocity_Verlet.pdf (2018/11/22)

		a                  = (2 - self.frictCoeff * self.time_step) / (2 + self.frictCoeff * self.time_step)
		b                  = np.sqrt(self.kb * self.temperature * self.time_step / 2)
		c                  = (2 * self.time_step) / (2 + (self.frictCoeff * self.time_step))
		random_eta         = np.random.normal(0, 1)

		current_force      = self.calForce(self.current_coord, self.current_vel)
		next_vel_half      = self.current_vel + (current_force * self.time_step / 2) + (b * random_eta)    # half time step
		next_coord         = self.current_coord + c * next_vel_half
		next_coord        -= (round(next_coord / self.boxsize_x) * self.boxsize_x) # PBC
		next_vel           = (a * next_vel_half) + (b * random_eta) + (self.time_step / 2) * current_force # full time_step
		next_time          = self.current_time + self.time_step 

		self.current_coord = next_coord
		self.current_vel   = next_vel
		self.current_time  = next_time
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

	# current_coord, current_time, time_step, time_length, fm, mass, boxsize_x, temperature, frictCoeff, abfCheckFlag, mode, fname_conventional, fname_force):
	# boxsize_x ranges from -pi ~ pi
	import sys
	#s = importanceSampling(0., 0., 0.005, 10, 0, 1., 6.283185307179586, 4, 1., "yes", "LangevinEngine", "wABF001.dat", "wABF_Force001.dat").mdrun()
	s = importanceSampling(0., 0., 0.005, float(sys.argv[3]), 0, 1., 6.283185307179586, 4, 1., "yes", "LangevinEngine", sys.argv[1], sys.argv[2]).mdrun()

