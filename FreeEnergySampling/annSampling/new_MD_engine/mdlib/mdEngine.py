#!/usr/bin/env python3
from mdlib.customMathFunc import myRound

class mdEngine(object):

	def __init__(self, nparticle, box, kb, time_step, temperature, ndims, mass, thermostatFlag="Langevin", frictCoeff=None):
		self.nparticle      = nparticle
		self.box            = box
		self.kb             = kb
		self.time_step      = time_step
		self.temperature    = temperature
		self.ndims          = ndims
		self.mass           = mass
		self.thermoStatFlag = thermoStatFlag 
		self.frictCoeff     = frictCoeff

	def velocityVerletSimple(self, current_coord, current_vel, getforce):

		if self.ndims == 1:
			for n in range(nparticle):
				current_force        = getForce(current_coord[n][0], 0, current_vel[n][0])

				current_coord[n][0] += current_vel[n][0] * time_step + (0.5 / mass) * current_force * time_step ** 2 
				cuurent_coord[n][0] -= (myRound(current_coord[n][0] / self.box[0]) * self.box[0])

				next_force           = getForce(current_coord[n][0], 0, current_vel[n][0]) 

				current_vel         += (0.5 / mass) * (current_force + next_force) * time_step

		if self.ndims == 2:
			for n in range(nparticle):
				current_force_x      = getForce(current_coord[n][0], 0, current_vel[n][0], current_coord[n][1])
				current_force_y      = getForce(current_coord[n][0], 1, current_vel[n][1], current_coord[n][1])

				current_coord[n][0] += current_vel[n][0] * time_step + (0.5 / mass) * current_force_x * time_step ** 2 
				current_coord[n][1] += current_vel[n][0] * time_step + (0.5 / mass) * current_force_y * time_step ** 2 
				cuurent_coord[n][0] -= (myRound(current_coord[n][0] / self.box[0]) * self.box[0])
				cuurent_coord[n][1] -= (myRound(current_coord[n][1] / self.box[1]) * self.box[1])

				next_force_x         = getForce(current_coord[n][0], 0, current_vel[n][0], current_coord[n][1]) 
				next_force_y         = getForce(current_coord[n][0], 1, current_vel[n][1], current_coord[n][1]) 

				current_vel[n][0]   += (0.5 / mass) * (current_force_x + next_force_x) * time_step
				current_vel[n][1]   += (0.5 / mass) * (current_force_y + next_force_y) * time_step

	def velocityVerletLJ(self, current_coord, current_vel, getForce):
		# for Lennard Jones potential
		pass

if __name__ == "__main__":
	pass
			
