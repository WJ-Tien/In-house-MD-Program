#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pickle

class trainingNN(object):
	
	def __init__(self):

		self.boltz_train         = []
		self.force_train         = []
		self.force_learn         = []
		
	def readData(self, file_boltzHistogram, file_forceHistogram): # should exist better framwork here 

		self.fileIn_boltz = open(str(file_boltzHistogram), "r") # 0 1
		self.fileIn_force = open(str(file_forceHistogram), "r") # 0 3

		for line_boltz in self.fileIn_boltz:
			self.boltz_train.append(float(line_boltz.split()[1]))

		for line_force in self.fileIn_force:
			self.force_train.append(float(line_force.split()[3]))

		self.fileIn_boltz.close()
		self.fileIn_force.close()
		
		# equalize the sample numbers for boltz and force 
		if len(self.boltz_train) == len(self.force_train):
			pass
		else:
			if len(self.boltz_train) > len(self.force_train):
				self.boltz_train.pop() 
			else:
				self.force_train.pop()

	def training(self, learning_rate):

		x_data = np.array(self.force_train)
		w1 = tf.Variable(tf.random_uniform([1], 1, 3))
		w2 = tf.Variable(tf.random_uniform([1], 1, 3))
		w3 = tf.Variable(tf.random_uniform([1], 1, 3))
		w4 = tf.Variable(tf.random_uniform([1], 1, 3))
		w5 = tf.Variable(tf.random_uniform([1], 1, 3))
		w6 = tf.Variable(tf.random_uniform([1], 1, 3))
		b = tf.Variable(tf.zeros([1]))
		y_real = np.sin(x_data) + 2*np.sin(2*x_data) + 3*np.sin(3*x_data) 
		y_predicted = w1 * tf.sin(w4*x_data) + w2 * tf.sin(w5*x_data) + w3 * tf.sin(w6*x_data) + b
		# cannot use np.sin here since it is not tensor-based
		# must use tf.sin

		# Minimize the mean squared errors.
		loss = tf.reduce_mean(tf.square(y_predicted - y_real))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train = optimizer.minimize(loss)

		# Before starting, initialize the variables.  We will 'run' this first.
		init = tf.global_variables_initializer()

		# Launch the graph.
		sess = tf.Session()
		sess.run(init)

		for step in range(500001):
			sess.run(train)
			if step % 100 == 0:
				print(step, sess.run(w1), sess.run(w2), sess.run(w3), sess.run(w4), sess.run(w5), sess.run(w6), sess.run(b), sess.run(loss))


	def saveData(self):
		pass
		# save data using pickle

if __name__ == "__main__":
	s = trainingNN()
	s.readData("wABFHistogram001.dat", "wABF_force001.dat")
	s.readData("wABFHistogram002.dat", "wABF_force002.dat")
	s.training(0.0001)
