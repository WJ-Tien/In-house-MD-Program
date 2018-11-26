#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np
import pickle

class trainingNN(object):
	
	def __init__(self, fileLoss):

		self.boltz_train = []
		self.force_train = []
		self.force_learn = []
		self.Loss_train = open(str(fileLoss), "w")
		
		
	def readData(self, file_boltzHistogram, file_forceHistogram): 

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

		# must shuffle the data before training 
		np.random.shuffle(self.force_train)
	
	def training(self, learning_rate):

		# assume 1 input layer; 1 hidden layer; 1 output layer; each with one neuron
		# activation function: sigmoid

		x_data = np.array(self.force_train)
		w = tf.Variable(tf.random_uniform([1], -1, 1))
		b = tf.Variable(tf.zeros([1]))
		y_real = np.sin(x_data) + 2*np.sin(2*x_data) + 3*np.sin(3*x_data) 
		y_predicted = tf.nn.sigmoid(w * x_data + b)
		# y_predicted = w1 * tf.sin(w4*x_data) + w2 * tf.sin(w5*x_data) + w3 * tf.sin(w6*x_data) + b

		loss = tf.reduce_mean(tf.square(y_predicted - y_real))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train = optimizer.minimize(loss)

		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)

		for step in range(500001):
			sess.run(train)
			if step % 50 == 0:
				print("Training Step %d" %(step))
				print("Weight %f"        %(sess.run(w)[0]))
				print("Bias %f"          %(sess.run(b)[0]))
				print("Loss %f"          %(sess.run(loss)))
				self.Loss_train.write(str(step) + " " + str(sess.run(loss)) + "\n")

		self.Loss_train.close()

	def saveData(self):

		# save training data using pickle
		pass

if __name__ == "__main__":
	
	s = trainingNN("analysis/loss.dat")
	s.readData("trainingSet/wABFHistogram003.dat", "trainingSet/wABF_Force003.dat")
	s.readData("trainingSet/wABFHistogram004.dat", "trainingSet/wABF_Force004.dat")
	s.readData("trainingSet/wABFHistogram005.dat", "trainingSet/wABF_Force005.dat")
	s.training(0.0001)
