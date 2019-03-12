#!/usr/bin/env python3
from mdlib.customMathFunc import myRound
import numpy as np

class mdEngine(object):

	def __init__(self, nparticle, box, kb, time_step, temperature, ndims, mass, thermoStatFlag, getForce, frictCoeff):
		self.nparticle      = nparticle
		self.box            = box
		self.kb             = kb
		self.time_step      = time_step
		self.temperature    = temperature
		self.ndims          = ndims
		self.mass           = mass
		self.thermoStatFlag = thermoStatFlag 
		self.frictCoeff     = frictCoeff
		self.getForce       = getForce
		self._velSQ         = 0.0
		self._count         = 0

	def genVelocity(self):	
		""" generate velocity profile by Maxwell-Boltzmann distribution"""
		velDirection = 1 if np.random.randint(1, 1001) % 2 == 0 else -1 
		initVel = np.ones((self.nparticle, self.ndims), dtype=np.float64) * velDirection * np.sqrt(self.kb * self.temperature / self.mass)
		return initVel

	def checkTargetTemperature(self, input_numpy_array_vel, current_frame, total_frame):
		self._count += 1
		for n in range(self.nparticle):
			for d in range(self.ndims):	
				self._velSQ += (input_numpy_array_vel[n][d]**2)

		if current_frame == total_frame:	
			targetTemperature = (self.mass * self._velSQ / self.kb / self.ndims / self._count)
			with open("checkTargetTemperaturem%.3fT%.3fgamma%.3f.dat" % (self.mass, self.temperature, self.frictCoeff), "w") as fout:
				fout.write("# Average Target Temperature: " + str(targetTemperature)) 

	def velocityVerletSimple(self, current_coord, current_vel):

		if self.ndims == 1:
			for n in range(self.nparticle):
				current_force        = self.getForce(current_coord[n][0], 0, current_vel[n][0], 0)

				current_coord[n][0]  = current_coord[n][0] + current_vel[n][0] * self.time_step + (0.5 / self.mass) * current_force * self.time_step**2 
				current_coord[n][0] -= (myRound(current_coord[n][0] / self.box[0]) * self.box[0])

				next_force           = self.getForce(current_coord[n][0], 0, current_vel[n][0], 0) 

				current_vel[n][0]    = current_vel[n][0] + (0.5 / self.mass) * (current_force + next_force) * self.time_step

		if self.ndims == 2:
			for n in range(self.nparticle):

				current_force_x      = self.getForce(current_coord[n][0], 0, current_vel[n][0], current_coord[n][1])
				current_force_y      = self.getForce(current_coord[n][0], 1, current_vel[n][1], current_coord[n][1])

				current_coord[n][0]  = current_coord[n][0] + current_vel[n][0] * self.time_step + (0.5 / self.mass) * current_force_x * self.time_step ** 2 
				current_coord[n][1]  = current_coord[n][1] + current_vel[n][1] * self.time_step + (0.5 / self.mass) * current_force_y * self.time_step ** 2 
				current_coord[n][0] -= (myRound(current_coord[n][0] / self.box[0]) * self.box[0])
				current_coord[n][1] -= (myRound(current_coord[n][1] / self.box[1]) * self.box[1])

				next_force_x         = self.getForce(current_coord[n][0], 0, current_vel[n][0], current_coord[n][1]) 
				next_force_y         = self.getForce(current_coord[n][0], 1, current_vel[n][1], current_coord[n][1]) 

				current_vel[n][0]    = current_vel[n][0] + (0.5 / self.mass) * (current_force_x + next_force_x) * self.time_step
				current_vel[n][1]    = current_vel[n][1] + (0.5 / self.mass) * (current_force_y + next_force_y) * self.time_step

	def velocityVerletLJ(self, current_coord, current_vel):
		# for Lennard Jones potential
		pass

if __name__ == "__main__":
	pass
			
