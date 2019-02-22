#!/usr/bin/env python3
from mdlib.mdEngine import mdEngine
from mdlib.mdFileIO import mdFileIO
from mdlib.force import Force
from mdlib.customMathFunc import getIndices
from annlib.ANN import trainingANN
import numpy as np
import tensorflow as tf


class ABF(object):

	def __init__(self, input_mdp_name):
		self.name            = input_mdp_name
		self.p               = mdFileIO().readParamFile(self.name) # p for md parameters
		self.bins            = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"], dtype=np.float64)
		self.colvars_coord   = np.linspace(-self.p["half_boxboundary"], self.p["half_boxboundary"], self.p["binNum"], dtype=np.float64)
		self.initializeForce = Force(self.p["kb"], self.p["time_step"], self.p["temperature"], self.p["ndims"], self.p["mass"], self.p["thermoStatFlag"], self.p["frictCoeff"])
		self.current_coord   = np.zeros((self.p["nparticle"], self.p["ndims"]), dtype=np.float64)
		self.velDirection    = 1 if np.random.randint(1, 1001) % 2 == 0 else -1 
		self.current_vel     = np.ones((self.p["nparticle"], self.p["ndims"]), dtype=np.float64) * self.velDirection * np.sqrt(self.p["kb"] * self.p["temperature"] / self.p["mass"])
		
		# TODO determine coord in another module

		if self.p["ndims"] == 1:
			
			self.colvars_force    = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_force_NN = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_count    = np.zeros(len(self.bins), dtype=np.float64) 

		if self.p["ndims"] == 2:
			self.colvars_force    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_force_NN = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_count    = np.zeros((self.p["ndims"], len(self.bins), len(self.bins)), dtype=np.float64) 

	def _histDistrRecord(self, coord_x, coord_y=None, d=None):

		if self.p["ndims"] == 1:
			self.colvars_count[getIndices(coord_x, self.bins)] += 1

		if self.p["ndims"] == 2:
			self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += 1

	def _forceDistrRecord(self, coord_x, updated_Fsys, coord_y=None, d=None):

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

	def _calBiasingForce(self, coord_x, coord_y=None, d=None):
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
		def _wrapper(coord_x, d=None, vel=None, coord_y=None):
			return func(coord_x, d, coord_y) + self.initializeForce.getForce(coord_x, d, vel, coord_y) 
		return _wrapper

	@_abfDecorator
	def getCurrentForce(self, coord_x, d=None, vel=None, coord_y=None): 

		if self.p["abfCheckFlag"] == "no" and self.p["nnCheckFlag"] == "no":
			Fabf = 0	

		if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "no":
			Fabf = _calBiasingForce(self, coord_x, coord_y=None, d=None)

		if self.p["abfCheckFlag"] == "yes" and self.p["nnCheckFlag"] == "yes":

			if self.p["init_frame"] < self.p["trainingFreq"]:
				Fabf = _calBiasingForce(self, coord_x, coord_y=None, d=None)

			else: # ANN takes over here

				tf.reset_default_graph()

				with tf.Session() as sess: # reload the previous training model
					Saver = tf.train.import_meta_graph("net" + str(self.p["ndims"]) + "D" +"/netSaver.ckpt.meta")
					Saver.restore(sess, tf.train.latest_checkpoint("net" + str(self.p["ndims"]) +"D/"))
					graph = tf.get_default_graph()
					#y_estimatedOP = graph.get_operation_by_name("criticalOP")  #get tensor with suffix :0
					layerOutput = graph.get_tensor_by_name("annOutput:0") 

					if self.p["ndims"] == 1:
						coord_x = np.array([coord_x])[:, np.newaxis]	
						CV      = graph.get_tensor_by_name("colvars:0") 
						Fabf    = sess.run(layerOutput, feed_dict={CV: coord_x}).reshape(self.p["ndims"])[d]

					if self.p["ndims"] == 2:
						coord_x = np.array([coord_x])[:, np.newaxis]	
						coord_y = np.array([coord_y])[:, np.newaxis]	
						CV_x    = graph.get_tensor_by_name("colvars_x:0") 
						CV_y    = graph.get_tensor_by_name("colvars_y:0") 
						Fabf    = sess.run(layerOutput, feed_dict={CV_x: coord_x, CV_y: coord_y}).reshape(self.p["ndims"])[d] 

				tf.reset_default_graph()

		currentFsys = self.initializeForce.getForce(coord_x, d, vel, coord_y)
		self.forceDistrRecord(coord_x, currentFsys, coord_y=None, d=None)
		self.histDistrRecord(coord_x, coord_y=None, d=None)

		return Fabf

		# post-collection

	def learningProxy(self):
		if self.p["nnCheckFlag"] == "yes":
			if self.p["init_frame"] % self.p["trainingFreq"] == 0 and self.p["init_frame"] != 0: 
				output = trainingANN("loss.dat", "hyperparam.dat", self.p["ndims"], len(self.bins)) 

				self.colvars_force = (self.colvars_force / self.colvars_count)
				self.colvars_force[np.isnan(self.colvars_force)] = 0 # 0/0 = nan n/0 = inf

				if self.p["init_frame"] < self.p["trainingFreq"] * self.p["switchSteps"]:
					self.colvars_force_NN = \
					output.training(self.colvars_coord, self.colvars_force, self.p["earlyLearningRate"], self.p["earlyRegularCoeff"], self.p["earlyEpoch"], self.p["nnOutputFreq"]) 
				else:
					self.colvars_force_NN = \
					output.training(self.colvars_coord, self.colvars_force, self.p["lateLearningRate"], self.p["lateRegularCoeff"], self.p["lateEpoch"], self.p["nnOutputFreq"]) 
	
				self.colvars_force = (self.colvars_force * self.colvars_count)

	def mdrun(self):

		mdInitializer = mdEngine(self.p["nparticle"], self.p["box"], self.p["kb"],\
                             self.p["time_step"], self.p["temperature"], self.p["ndims"],\
                             self.p["mass"], self.p["thermostatFlag"], self.p["frictCoeff"])

		# the first frame
		mdFileIO().writeParams(self.p)
		self.conventionalDataOutput()
		self.printIt()
		
		# the rest of the frames
		while self.p["init_frame"] < self.p["total_frame"]: 
			if self.mode == "LangevinEngine":
				self.LangevinEngine()
				self.conventionalDataOutput()		
				self.printIt()

		self.forceOnColvarsOutput()
		self.closeAllFiles()
		pass	



if __name__ == "__main__":
	ABF("in.mdp")

	

	



