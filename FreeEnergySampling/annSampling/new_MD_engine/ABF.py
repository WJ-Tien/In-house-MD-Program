#!/usr/bin/env python3
from mdlib.mdEngine import mdEngine
from mdlib.mdFileIO import mdFileIO 
from mdlib.force import Force
from mdlib.customMathFunc import getIndices, paddingRighMostBins
from mdlib.render import rendering
from annlib.abfANN import trainingANN
import numpy as np
import tensorflow as tf
import time

class ABF(object):

	def __init__(self, input_mdp_file):

		self.IO              = mdFileIO() 
		self.p               = self.IO.readParamFile(input_mdp_file) # p for md parameters
		self.bins            = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"] + 1, dtype=np.float64)
		self.colvars_coord   = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"] + 1, dtype=np.float64)

		self.mdInitializer   = mdEngine(self.p["nparticle"], self.p["box"], self.p["kb"],\
																		 self.p["time_step"], self.p["temperature"], self.p["ndims"],\
																		 self.p["mass"], self.p["thermoStatFlag"], self.getCurrentForce, self.p["frictCoeff"])

		self.initializeForce = Force(self.p["kb"], self.p["time_step"], self.p["temperature"], self.p["ndims"], self.p["mass"], self.p["thermoStatFlag"], self.p["frictCoeff"])

		# TODO initialize atom_coords in another module 
		self.current_coord	 = np.zeros((self.p["nparticle"], self.p["ndims"]), dtype=np.float64)
		self.current_vel		 = self.mdInitializer.genVelocity() 
		
		if self.p["ndims"] == 1:
			self.colvars_force    = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_force_NN = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_count    = np.zeros(len(self.bins), dtype=np.float64) 

		if self.p["ndims"] == 2:
			self.colvars_force    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_force_NN = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_count    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 

	def _histDistrRecord(self, coord_x, coord_y, d):

		if self.p["ndims"] == 1:
			self.colvars_count[getIndices(coord_x, self.bins)] += 1

		if self.p["ndims"] == 2:
			self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += 1

	def _forceDistrRecord(self, coord_x, updated_Fsys, coord_y, d):

		if self.p["ndims"] == 1:
			self.colvars_force[getIndices(coord_x, self.bins)] += updated_Fsys 

		if self.p["ndims"] == 2:
			self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += updated_Fsys 

	def _inverseGradient(self):
		""" cv == cartesian so return 1"""
		return 1

	def _Jacobian(self):
		""" cv == cartesian -> ln|J| = 0 so return 0"""
		return 0

	def _entropicCorrection(self):
		return self.p["kb"] * self.p["temperature"] * self._Jacobian()

	def _calBiasingForce(self, coord_x, coord_y, d):
		if self.p["ndims"] == 1:
			if self.colvars_count[getIndices(coord_x, self.bins)] == 0:
				return 0
			else:
				return -((self.colvars_force[getIndices(coord_x, self.bins)] / self.colvars_count[getIndices(coord_x, self.bins)] + self._entropicCorrection()) * self._inverseGradient())

		if self.p["ndims"] == 2:
			if self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] == 0:
				return 0
			else:
				return -((self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] / self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] +\
								self._entropicCorrection()) * self._inverseGradient()) 

	def _abfDecorator(func):
		def _wrapper(self, coord_x, d, vel, coord_y):
			Fabf = func(self, coord_x, d, vel, coord_y)
			currentFsys = self.initializeForce.getForce(coord_x, d, vel, coord_y)
			self._forceDistrRecord(coord_x, currentFsys, coord_y, d)
			self._histDistrRecord(coord_x, coord_y, d)
			return Fabf + currentFsys # Fabf + currentFsys
		return _wrapper

	@_abfDecorator
	def getCurrentForce(self, coord_x, d, vel, coord_y): 

		if self.p["abfCheckFlag"] == "no" and self.p["nnCheckFlag"] == "no":
			Fabf = 0	

		elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "no":
			Fabf = self._calBiasingForce(coord_x, coord_y, d)

		elif self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":

			if self.p["init_frame"] < self.p["trainingFreq"]:
				Fabf = self._calBiasingForce(coord_x, coord_y, d)

			else: # ANN takes over here

				tf.reset_default_graph()

				with tf.Session() as sess: # reload the previous training model
					Saver = tf.train.import_meta_graph("net" + str(self.p["ndims"]) + "D" +"/netSaver.ckpt.meta")
					Saver.restore(sess, tf.train.latest_checkpoint("net" + str(self.p["ndims"]) +"D/"))
					graph = tf.get_default_graph()
					#y_estimatedOP = graph.get_operation_by_name("criticalOP")	#get tensor with suffix :0
					layerOutput = graph.get_tensor_by_name("annOutput:0") 

					if self.p["ndims"] == 1:
						coord_x = np.array([coord_x])[:, np.newaxis]	
						CV			= graph.get_tensor_by_name("colvars:0") 
						Fabf		= sess.run(layerOutput, feed_dict={CV: coord_x}).reshape(self.p["ndims"])[d]

					if self.p["ndims"] == 2:
						coord_x = np.array([coord_x])[:, np.newaxis]	
						coord_y = np.array([coord_y])[:, np.newaxis]	
						CV_x		= graph.get_tensor_by_name("colvars_x:0") 
						CV_y		= graph.get_tensor_by_name("colvars_y:0") 
						Fabf		= sess.run(layerOutput, feed_dict={CV_x: coord_x, CV_y: coord_y}).reshape(self.p["ndims"])[d] 

				tf.reset_default_graph()

		return Fabf

	def _learningProxy(self):
		if self.p["nnCheckFlag"] == "yes":
			if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0: 
				output = trainingANN("loss.dat", "hyperparam.dat", self.p["ndims"], len(self.bins)) 

				self.colvars_force = (self.colvars_force / self.colvars_count)
				self.colvars_force[np.isnan(self.colvars_force)] = 0 # 0/0 = nan n/0 = inf
				self.colvars_force = paddingRighMostBins(self.p["ndims"], self.colvars_force)

				if self.p["init_frame"] < self.p["trainingFreq"] * self.p["switchSteps"]:
					self.colvars_force_NN = \
					output.training(self.colvars_coord, self.colvars_force, self.p["earlyLearningRate"], self.p["earlyRegularCoeff"], self.p["earlyEpoch"], self.p["nnOutputFreq"]) 
				else:
					self.colvars_force_NN = \
					output.training(self.colvars_coord, self.colvars_force, self.p["lateLearningRate"], self.p["lateRegularCoeff"], self.p["lateEpoch"], self.p["nnOutputFreq"]) 
	
				self.colvars_force = (self.colvars_force * self.colvars_count)

	def mdrun(self):

		init_real_world_time = time.time()

		# pre-processing
		lammpstrj      = open("m%.1f_T%.3f_gamma%.4f_len_%d_%s_%s.lammpstrj" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"], self.p["abfCheckFlag"], self.p["nnCheckFlag"]), "w")
		forceOnCVs     = open("Force_m%.1fT%.3f_gamma%.4f_len_%d_%s_%s.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"], self.p["abfCheckFlag"], self.p["nnCheckFlag"]), "w")
		histogramOnCVs = open("Hist_m%.1fT%.3f_gamma%.4f_len_%d_%s_%s.dat" %(self.p["mass"], self.p["temperature"], self.p["frictCoeff"], self.p["total_frame"], self.p["abfCheckFlag"], self.p["nnCheckFlag"]), "w")

		# Start of the simulation
		# the first frame
		self.IO.writeParams(self.p)
		self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
		self.IO.printCurrentStatus(self.p["init_frame"], init_real_world_time)	
		
		# the rest of the frames
		while self.p["init_frame"] < self.p["total_frame"]: 

			self.p["init_frame"] += 1
			self.IO.printCurrentStatus(self.p["init_frame"], init_real_world_time)	

			self.mdInitializer.checkTargetTemperature(self.current_vel, self.p["init_frame"], self.p["total_frame"])

			if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0 and self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":
				self._learningProxy()
			
			self.mdInitializer.velocityVerletSimple(self.current_coord, self.current_vel)	

			self.IO.lammpsFormatColvarsOutput(self.p["ndims"], self.p["nparticle"], self.p["half_boxboundary"], self.p["init_frame"], self.current_coord, lammpstrj, self.p["writeFreq"]) 
		# End of simulation

		# post-processing
		probability = self.colvars_count / (np.sum(self.colvars_count) / self.p["ndims"]) # both numerator and denominator should actually be divided by two but this would be cacncelled
		probability = paddingRighMostBins(self.p["ndims"], probability) 
		self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, probability, self.colvars_count/2, histogramOnCVs)

		# original data output
		if self.p["nnCheckFlag"] == "yes":
			self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, self.colvars_force_NN, self.colvars_count, forceOnCVs)

		else:
			self.colvars_force = (self.colvars_force / self.colvars_count)
			self.colvars_force[np.isnan(self.colvars_force)] = 0
			self.colvars_force = paddingRighMostBins(self.p["ndims"], self.colvars_force)	
			self.IO.propertyOnColvarsOutput(self.p["ndims"], self.bins, self.colvars_force, self.colvars_count, forceOnCVs)

		# ndims >= 2 -> plot using matplotlib #TODO put it in the self.IO 
		if self.p["ndims"] == 2: 
			s = rendering(self.p["ndims"], self.p["half_boxboundary"], self.p["binNum"], self.p["temperature"])
			s.render(probability[0], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "boltzDist" +str(self.p["ndims"])+"D"))
			if self.p["nnCheckFlag"] == "yes":
				s.render(self.colvars_force_NN[0], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcex" +str(self.p["ndims"])+"D"))
				s.render(self.colvars_force_NN[1], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcey" +str(self.p["ndims"])+"D"))
			else:
				s.render(self.colvars_force[0], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcex" +str(self.p["ndims"])+"D"))
				s.render(self.colvars_force[1], name=str(self.p["abfCheckFlag"] + "_" + self.p["nnCheckFlag"] + "_" + "forcey" +str(self.p["ndims"])+"D"))

		self.IO.closeAllFiles(lammpstrj, forceOnCVs, histogramOnCVs)

if __name__ == "__main__":
	ABF("in.mdp").mdrun()
