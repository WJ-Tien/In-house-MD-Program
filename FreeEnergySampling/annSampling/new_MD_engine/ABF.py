#!/usr/bin/env python3
from mdlib.mdFileIO import mdFileIO
from mdlib.force import Force
from mdlib.mdEngine import mdEngine
import numpy as np

class ABF(object):

	def __init__(self, input_mdp_name):
		self.name = input_mdp_name
		p = mdFileIO().readParamFile(self.name) # p for md parameters
		self.bins = np.linspace(-p["half_boxboundary"], p["half_boxboundary"], p["binNum"], dtype=np.float64)
		
		self.colvars_coord = np.linspace(-p["half_boxboundary"], p["half_boxboundary"], p["binNum"], dtype=np.float64)
		
		# TODO determine coord in another module
		self.current_coord = np.zeros((p["nparticle"], p["ndims"]), dtype=np.float64)
		self.velDirection = 1 if np.random.randint(1, 1001) % 2 == 0 else -1 
		self.current_vel = np.ones((p["nparticle"], p["ndims"]), dtype=np.float64) * \
                       self.velDirection * np.sqrt(p["kb"] * p["temperature"] / p["mass"])
		#filein
		#fileout
		if p["ndims"] == 1:
			
			self.colvars_force    = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_force_NN = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_count    = np.zeros(len(self.bins), dtype=np.float64) 

		if p["ndims"] == 2:
			self.colvars_force    = np.zeros((self.ndims, len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_force_NN = np.zeros((self.ndims, len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_count    = np.zeros((self.ndims, len(self.bins), len(self.bins)), dtype=np.float64) 
	def abf_getForce(self): #TODO decorator
		pass

	def forceDistrRecord(self):
		pass


	def histDistrRecord(self):
		pass
	

if __name__ == "__main__":
	ABF("in.mdp")

	

	



