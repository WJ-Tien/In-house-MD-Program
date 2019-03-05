#!/usr/bin/env python3
import numpy as np
from mdlib.customMathFunc import randMars

class Force(object):

	def __init__(self, kb, time_step, temperature, ndims, mass, thermoStatFlag, frictCoeff, sigma=None, epsilon=None):
		self.kb							= kb
		self.time_step			= time_step
		self.temperature		= temperature
		self.ndims					= ndims
		self.mass						= mass
		self.thermoStatFlag = thermoStatFlag 
		self.frictCoeff			= frictCoeff
		self.sigma          = sigma
		self.epsilon        = epsilon

	def _potentialForceSimple(self, coord_x, coord_y, d):			

		# For 1D toy model
		# Potential surface of the system: cosx + cos2x + cos3x
		# np.cos(CartesianCoord) + np.cos(2*CartesianCoord) + np.cos(3*CartesianCoord)
		# Reaction Coordinate (colvars) == Cartesian Coordinate, so Jacobian J = 1

		# For 2D toy model
		# Potential surface of the system: (0.0011- x*0.421 + x**4 + 2*x**3 + 3*y + y**3 + y**2 + x*2) * exp(-x**2 - y**2) 
		# Reaction Coordinate (colvars) == Cartesian Coordinate, so Jacobian J = 1

		if self.ndims == 1: # force along x
			return np.sin(coord_x) + 2*np.sin(2*coord_x) + 3*np.sin(3*coord_x)

		if self.ndims == 2:
			if d == 0:				# force along x	
				return -(-2*coord_x*(coord_x**4 + 2*coord_x**3 + 1.579*coord_x + coord_y**3 + coord_y**2 + 3*coord_y + 0.0011)*np.exp(-coord_x**2 - coord_y**2) +\
							 (4*coord_x**3 + 6*coord_x**2 + 1.579)*np.exp(-coord_x**2 - coord_y**2))

			if d == 1:				# force along y	
				return -(-2*coord_y*(coord_x**4 + 2*coord_x**3 + 1.579*coord_x + coord_y**3 + coord_y**2 + 3*coord_y + 0.0011)*np.exp(-coord_x**2 - coord_y**2) +\
							 (3*coord_y**2 + 2*coord_y + 3)*np.exp(-coord_x**2 - coord_y**2))

	def _potentialForceLJ(self, r):
		return 4 * self.epsilon * self.sigma**6 * (self.sigma**6 * -12 * r**(-13) + 6 * self.sigma**(-7))

	def _viscousForce(self, vel):
		return -self.frictCoeff * vel * self.mass

	def _randomForce(self): 
		gaussian_random_value = np.random.normal(0, 1)
		return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature / self.time_step) * (gaussian_random_value)
		#uniform_random_value = np.random.uniform(-np.sqrt(3), np.sqrt(3)) 
		#return np.sqrt(24 * self.mass * self.frictCoeff * self.kb * self.temperature / self.time_step) * (uniform_random_value)
		#randMars_uniform_random_value = randMars() - 0.5
		#return np.sqrt(24 * self.mass * self.frictCoeff * self.kb * self.temperature / self.time_step) * (randMars_uniform_random_value)
		#uniform_random_value = np.random.uniform(-np.sqrt(3), np.sqrt(3)) 
		#return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature / self.time_step) * (uniform_random_value)

	def getForce(self, coord_x, d, vel, coord_y):

		if self.thermoStatFlag == "newton":
			if self.ndims == 1:
				return self._potentialForceSimple(coord_x, 0, 0) 
			if self.ndims == 2:
				return self._potentialForceSimple(coord_x, coord_y, d) 

		if self.thermoStatFlag == "langevin":
			if self.ndims == 1:
				return self._potentialForceSimple(coord_x, 0, 0) + self._viscousForce(vel) + self._randomForce()
			if self.ndims == 2:
				return self._potentialForceSimple(coord_x, coord_y, d) + self._viscousForce(vel) + self._randomForce()
		
if __name__ == "__main__":
	pass
