#!/usr/bin/env python3
import numpy as np
import time
from ANN import trainingANN 
from customMathFunc import myRound, getIndices, truncateFloat
import tensorflow as tf

class ABF(object):
	
	def __init__(self, particles, ndims, current_time, time_step, time_length, frame, mass, box, temperature, frictCoeff, abfCheckFlag, nnCheckFlag, trainingFrequency, mode, learningRate, regularCoeff, epoch, lateLearningRate, lateRegularCoeff,lateEpoch,switchSteps, NNoutputFreq, half_boxboundary, binNum, filename_conventional, filename_force):

		self.startTime        = time.time()
		self.particles        = particles
		self.ndims            = ndims
		self.kb               = 1   
		self.binNum           = binNum 
		self.half_boxboundary = half_boxboundary
		self.bins             = np.linspace(-half_boxboundary, half_boxboundary, self.binNum, dtype=np.float64)
		self.colvars_coord    = np.linspace(-half_boxboundary, half_boxboundary, self.binNum, dtype=np.float64) # should discard the last one 
		self.current_coord    = np.zeros((self.particles, self.ndims), dtype=np.float64) 
		self.current_time     = current_time
		self.time_step        = time_step
		self.time_length      = time_length
		self.frame            = frame
		self.mass             = mass 
		self.box              = np.array(box)  
		self.temperature      = temperature
		self.pve_or_nve       = 1 if np.random.randint(1, 1001) % 2 == 0 else	-1
		self.current_vel      = np.array([[self.pve_or_nve * (np.sqrt(self.kb * self.temperature / self.mass)) for j in range(self.ndims)] for i in range(self.particles)], dtype=np.float64) 
		self.frictCoeff       = frictCoeff
		self.mode             = mode
		self.abfCheckFlag     = abfCheckFlag 
		self.nnCheckFlag      = nnCheckFlag 
		self.Frequency        = trainingFrequency
		self.fileOut          = open(str(filename_conventional), "w") 
		self.fileOutForce     = open(str(filename_force), "w") 
		self.learningRate     = learningRate 
		self.regularCoeff     = regularCoeff 
		self.epoch            = epoch 
		self.lateLearningRate = lateLearningRate 
		self.regularCoeff     = lateRegularCoeff 
		self.epoch            = lateEpoch 
		self.switchSteps      = switchSteps
		self.NNoutputFreq     = NNoutputFreq 

		if self.ndims == 1:
			self.colvars_force    = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_force_NN = np.zeros(len(self.bins), dtype=np.float64) 
			self.colvars_count    = np.zeros(len(self.bins), dtype=np.float64) 

		if self.ndims == 2:
			self.colvars_force    = np.zeros((self.ndims, len(self.bins), len(self.bins)), dtype=np.float64) # ixj
			self.colvars_force_NN = np.zeros((self.ndims, len(self.bins), len(self.bins)), dtype=np.float64) 
			self.colvars_count    = np.zeros((self.ndims, len(self.bins), len(self.bins)), dtype=np.float64) 

	def printIt(self):
		print("Frame %d with time %f" % (self.frame, time.time() - self.startTime)) 

	def writeFileHead(self):
		self.fileOut.write("#" + " " + "Mode"         + " " + str(self.mode)         + "\n")
		self.fileOut.write("#" + " " + "Dims"         + " " + str(self.ndims)        + "\n")
		self.fileOut.write("#" + " " + "Particles"    + " " + str(self.particles)    + "\n")
		self.fileOut.write("#" + " " + "BinNumber"    + " " + str(self.binNum)       + "\n")
		self.fileOut.write("#" + " " + "Temperature"  + " " + str(self.temperature)  + "\n") 
		self.fileOut.write("#" + " " + "FrictCoeff"   + " " + str(self.frictCoeff)   + "\n") 
		self.fileOut.write("#" + " " + "Time_length"  + " " + str(self.time_length)  + "\n") 
		self.fileOut.write("#" + " " + "Time_step"    + " " + str(self.time_step)    + "\n") 
		self.fileOut.write("#" + " " + "abfCheckFlag" + " " + str(self.abfCheckFlag) + "\n")
		self.fileOut.write("#" + " " + "nnCheckFlag"  + " " + str(self.nnCheckFlag)  + "\n")
		self.fileOut.write("#" + " " + "Frame" + " "  + "Time" + " " + "Coordinates(xyz)" + " " + "Velocity(xyz)" + "\n")

	def conventionalDataOutput(self):
		for n in range(self.particles):	
			self.fileOut.write(str(self.frame) + " " + str(self.current_time) + " ")
			for d in range(self.ndims):
				self.fileOut.write(str(self.current_coord[n][d]) + " " + str(self.current_vel[n][d]) + " ") 
			self.fileOut.write("\n")

	def forceOnColvarsOutput(self): 
		
		self.colvars_force = (self.colvars_force / self.colvars_count)
		self.colvars_force[np.isnan(self.colvars_force)] = 0

		if self.nnCheckFlag == "yes" and self.abfCheckFlag == "yes":
			if self.ndims == 1:
				for i in range(len(self.bins)): 
					self.fileOutForce.write(str(self.colvars_coord[i]) + " ")
					self.fileOutForce.write(str(self.colvars_force_NN[i]) + " " + str(self.colvars_count[i]) + "\n")  

			if self.ndims == 2:
				for i in range(len(self.bins)):
					for j in range(len(self.bins)):
						self.fileOutForce.write(str(self.colvars_coord[i]) + " ")
						self.fileOutForce.write(str(self.colvars_coord[j]) + " ")
						self.fileOutForce.write(str(self.colvars_force_NN[0][i][j]) + " " + str(self.colvars_count[0][i][j]) + " " +str(self.colvars_force_NN[1][i][j]) + " " + str(self.colvars_count[1][i][j]) + "\n")  
		else:
			if self.ndims == 1:
				for i in range(len(self.bins)): 
					self.fileOutForce.write(str(self.colvars_coord[i]) + " ")
					self.fileOutForce.write(str(self.colvars_force[i]) + " " + str(self.colvars_count[i]) + "\n")  

			if self.ndims == 2:
				for i in range(len(self.bins)):
					for j in range(len(self.bins)):
						self.fileOutForce.write(str(self.colvars_coord[i]) + " ")
						self.fileOutForce.write(str(self.colvars_coord[j]) + " ")
						self.fileOutForce.write(str(self.colvars_force[0][i][j]) + " " + str(self.colvars_count[0][i][j]) + " " +str(self.colvars_force[1][i][j]) + " " + str(self.colvars_count[1][i][j]) + "\n")  

	def PotentialForce(self, coord_x, coord_y, d):     

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
			if d == 0:        # force along x	
				return -(-2*coord_x*(coord_x**4 + 2*coord_x**3 + 1.579*coord_x + coord_y**3 + coord_y**2 + 3*coord_y + 0.0011)*np.exp(-coord_x**2 - coord_y**2) +\
               (4*coord_x**3 + 6*coord_x**2 + 1.579)*np.exp(-coord_x**2 - coord_y**2))

			if d == 1:        # force along y	
				return -(-2*coord_y*(coord_x**4 + 2*coord_x**3 + 1.579*coord_x + coord_y**3 + coord_y**2 + 3*coord_y + 0.0011)*np.exp(-coord_x**2 - coord_y**2) +\
               (3*coord_y**2 + 2*coord_y + 3)*np.exp(-coord_x**2 - coord_y**2))

	def visForce(self, vel):
		return -self.frictCoeff * vel * self.mass

	def randForce(self): 
		random_constant = np.random.normal(0, 1)
		return np.sqrt(2 * self.mass * self.frictCoeff * self.kb * self.temperature / self.time_step) * (random_constant)

	def appliedBiasForce(self, coord_x, coord_y, d):

		if self.ndims == 1:
			if self.colvars_count[getIndices(coord_x, self.bins)] == 0:
				return 0
			else:
				return -(self.colvars_force[getIndices(coord_x, self.bins)] / self.colvars_count[getIndices(coord_x, self.bins)]) 

		if self.ndims == 2:
			if self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] == 0:
				return 0
			else:
				return -(self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] / self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)]) 

	def forceDistrRecord(self, coord_x, coord_y, updated_Fsys, d):

		if self.ndims == 1:
			if isinstance(updated_Fsys, float) or isinstance(updated_Fsys, int): # conventional ABF collection, which is a float
				self.colvars_force[getIndices(coord_x, self.bins)] += updated_Fsys 
				self.colvars_count[getIndices(coord_x, self.bins)] += 1
			else:                                       # refined force from NN, which is a np.ndarray
				self.colvars_force += updated_Fsys
				self.colvars_count += 1

		if self.ndims == 2:
			if isinstance(updated_Fsys, float) or isinstance(updated_Fsys, int):
				self.colvars_force[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += updated_Fsys 
				self.colvars_count[d][getIndices(coord_x, self.bins)][getIndices(coord_y, self.bins)] += 1
			else: 
				self.colvars_force += updated_Fsys
				self.colvars_count += 1

	def learningProxy(self):
		if self.nnCheckFlag == "yes":
			if self.frame % self.Frequency == 0 and self.frame != 0: 
				output = trainingANN("loss.dat", "hyperparam.dat", self.ndims, len(self.bins)) 

				self.colvars_force = (self.colvars_force / self.colvars_count)
				self.colvars_force[np.isnan(self.colvars_force)] = 0 # 0/0 = nan n/0 = inf
				if self.frame > self.Frequency * self.switchSteps:
					self.learningRate = lateLearningRate 
					self.regularCoeff = lateRegularCoeff 
					self.epoch        = lateEpoch 

				self.colvars_force_NN = \
				output.training(self.colvars_coord, self.colvars_force, self.learningRate, self.regularCoeff, self.epoch, self.NNoutputFreq) 
	
				self.colvars_force = (self.colvars_force * self.colvars_count)

	def getLocalForce(self, coord_x, coord_y, vel, d=None):

		coord_x = truncateFloat(coord_x)
		coord_y = truncateFloat(coord_y)
		Fu = self.PotentialForce(coord_x, coord_y, d) 
		Fsys = self.PotentialForce(coord_x, coord_y, d) + self.visForce(vel) + self.randForce()

		# Regular MD
		if self.abfCheckFlag == "no" and self.nnCheckFlag == "no":  
			self.forceDistrRecord(coord_x, coord_y, Fsys, d) 
			return Fu / self.mass 

	  # Regular MD with ABF
		if self.abfCheckFlag == "yes" and self.nnCheckFlag == "no":
			Fabf = self.appliedBiasForce(coord_x, coord_y, d)
			self.forceDistrRecord(coord_x, coord_y, Fsys, d) 
			return (Fu + Fabf) / self.mass

		# Regular MD with ABF and ANN (I)
		if self.abfCheckFlag == "yes" and self.nnCheckFlag == "yes": 
			if self.frame < self.Frequency: 
				Fabf = self.appliedBiasForce(coord_x, coord_y, d)
				self.forceDistrRecord(coord_x, coord_y, Fsys, d) 
				return (Fu + Fabf) / self.mass
			else: # NN takes over here

				self.forceDistrRecord(coord_x, coord_y, Fsys, d) 
				tf.reset_default_graph()

				with tf.Session() as sess: # reload the previous training model
					Saver = tf.train.import_meta_graph("net1D/netSaver.ckpt.meta")
					Saver.restore(sess, tf.train.latest_checkpoint("net1D/"))
					graph = tf.get_default_graph()
					#y_estimatedOP = graph.get_operation_by_name("criticalOP") 
					layerOutput = graph.get_tensor_by_name("annOutput:0") 

					if self.ndims == 1:
						coord_x = np.array([coord_x])[:, np.newaxis]	
						CV = graph.get_tensor_by_name("colvars:0") 
						Fabf = sess.run(layerOutput, feed_dict={CV: coord_x}).reshape(self.particles)[0]
					if self.ndims == 2:
						coord_x = np.array([coord_x])[:, np.newaxis]	
						coord_y = np.array([coord_y])[:, np.newaxis]	
						CV_x = graph.get_tensor_by_name("colvars_x:0") 
						CV_y = graph.get_tensor_by_name("colvars_y:0") 
						Fabf = sess.run(layerOutput, feed_dict={CV_x: coord_x, CV_y: coord_y}).reshape(self.particles)[0]

				tf.reset_default_graph()

				return (Fu + Fabf) / self.mass

	def LangevinEngine(self):

		# molecular_dynamics_2015.pdf
		# http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
		# https://pdfs.semanticscholar.org/f393/85336df44c2af1fd6f293540b18a701b1c56.pdf


		if self.ndims == 1:
			random_xi_x       = np.random.normal(0, 1)
			random_theta_x    = np.random.normal(0, 1)
			self.learningProxy()

			for n in range(self.particles):

				sigma                       = np.sqrt(2 * self.kb * self.temperature * self.frictCoeff / self.mass)

				current_force_x             = self.getLocalForce(self.current_coord[n][0], 0, self.current_vel[n][0], 0) 

				Ct_x                        = (0.5 * self.time_step**2) * (current_force_x - self.frictCoeff * self.current_vel[n][0]) + \
																			sigma * (self.time_step**1.5) * (0.5 * random_xi_x + (np.sqrt(3)/6) * random_theta_x)

				self.current_coord[n][0]    = self.current_coord[n][0] + (self.time_step * self.current_vel[n][0]) + Ct_x
				self.current_coord[n][0]   -= (myRound(self.current_coord[n][0] / self.box[0]) * self.box[0]) # PBC

				updated_force_x             = self.getLocalForce(self.current_coord[n][0], 0, self.current_vel[n][0], 0) 

				self.current_vel[n][0]      = self.current_vel[n][0] + (0.5 * self.time_step * (current_force_x + updated_force_x)) - (self.time_step * self.frictCoeff * self.current_vel[n][0]) + \
																			(np.sqrt(self.time_step) * sigma * random_xi_x) - (self.frictCoeff * Ct_x)
				

		if self.ndims == 2:

			random_xi_x       = np.random.normal(0, 1)
			random_theta_x    = np.random.normal(0, 1)
			random_xi_y       = np.random.normal(0, 1)
			random_theta_y    = np.random.normal(0, 1)
			self.learningProxy()

			for n in range(self.particles):

				sigma                       = np.sqrt(2 * self.kb * self.temperature * self.frictCoeff / self.mass)

				current_force_x             = self.getLocalForce(self.current_coord[n][0], self.current_coord[n][1], self.current_vel[n][0], 0) 
				current_force_y             = self.getLocalForce(self.current_coord[n][0], self.current_coord[n][1], self.current_vel[n][1], 1) 

				Ct_x                        = (0.5 * self.time_step**2) * (current_force_x - self.frictCoeff * self.current_vel[n][0]) + \
																			sigma * (self.time_step**1.5) * (0.5 * random_xi_x + (np.sqrt(3)/6) * random_theta_x)

				Ct_y                        = (0.5 * self.time_step**2) * (current_force_y - self.frictCoeff * self.current_vel[n][1]) + \
																			sigma * (self.time_step**1.5) * (0.5 * random_xi_y + (np.sqrt(3)/6) * random_theta_y)

				self.current_coord[n][0]    = self.current_coord[n][0] + (self.time_step * self.current_vel[n][0]) + Ct_x
				self.current_coord[n][0]   -= (myRound(self.current_coord[n][0] / self.box[0]) * self.box[0]) 
				self.current_coord[n][1]    = self.current_coord[n][1] + (self.time_step * self.current_vel[n][1]) + Ct_y
				self.current_coord[n][1]   -= (myRound(self.current_coord[n][1] / self.box[1]) * self.box[1]) 

				updated_force_x             = self.getLocalForce(self.current_coord[n][0], self.current_coord[n][1], self.current_vel[n][0], 0) 
				updated_force_y             = self.getLocalForce(self.current_coord[n][0], self.current_coord[n][1], self.current_vel[n][1], 1) 

				self.current_vel[n][0]      = self.current_vel[n][0] + (0.5 * self.time_step * (current_force_x + updated_force_x)) - (self.time_step * self.frictCoeff * self.current_vel[n][0]) + \
																			(np.sqrt(self.time_step) * sigma * random_xi_x) - (self.frictCoeff * Ct_x)
				self.current_vel[n][1]      = self.current_vel[n][1] + (0.5 * self.time_step * (current_force_y + updated_force_y)) - (self.time_step * self.frictCoeff * self.current_vel[n][1]) + \
																			(np.sqrt(self.time_step) * sigma * random_xi_y) - (self.frictCoeff * Ct_y)

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
	pass	

