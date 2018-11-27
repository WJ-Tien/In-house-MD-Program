#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np
import pickle

class trainingNN(object):
	
	def __init__(self, fileLoss, filehyperparam, fileforceResult):

		self.force_to_train  = []
		self.force_to_learn  = []
		self.cv              = []
		self.Loss_train      = open(str(fileLoss), "w")
		self.hp_train        = open(str(filehyperparam), "w")
		self.force_result    = open(str(fileforceResult), "w")
		
	def readData(self, file_boltzHistogram, file_force_to_train, file_force_to_learn): 

		self.fileIn_force_to_train = open(str(file_force_to_train), "r") # 0 3
		self.fileIn_force_to_learn = open(str(file_force_to_learn), "r") # 0 3


		for line_force_to_train in self.fileIn_force_to_train:
			self.cv.append(float(line_force_to_train.split()[0]))
			self.force_to_train.append(float(line_force_to_train.split()[3]))
		
		for line_force_to_learn in self.fileIn_force_to_learn:
			self.force_to_learn.append(float(line_force_to_learn.split()[3]))

		self.fileIn_force_to_train.close()
		self.fileIn_force_to_learn.close()
		
		# equalize the sample numbers for training and learning array
		if len(self.force_to_train) == len(self.force_to_learn):
			pass
		else:
			while(len(self.force_to_train) > len(self.force_to_learn)):
				self.force_to_train.pop() 
				self.cv.pop() 
			while(len(self.force_to_train) < len(self.force_to_learn)):
				self.force_to_learn.pop()

		# must shuffle the data before training 
		#np.random.shuffle(self.force_to_train)
	
	def training(self, learning_rate, step, outputFreq):

		# assume 1 input layer; 1 hidden layer; 1 output layer; each with one neuron
		# activation function: sigmoid

		# data initialization
		self.cv     = np.array(self.cv)
		x_data      = np.array(self.force_to_learn)
		w           = tf.Variable(tf.random_uniform([len(self.force_to_train)], -1, 1))
		b           = tf.Variable(tf.zeros([len(self.force_to_train)]))
		y_real      = np.array(self.force_to_train)  
		y_predicted = tf.nn.relu(w * x_data + b)

		# define loss function and associated optimizer
		#loss = tf.reduce_sum(tf.square(y_predicted - y_real))
		loss        = tf.reduce_mean(tf.square(y_predicted - y_real))
		optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
		train       = optimizer.minimize(loss)

		# initialize the "tensorflow graph"
		init = tf.global_variables_initializer()

		# run the graph (session)
		sess = tf.Session()
		sess.run(init)

		for step in range(step):
			sess.run(train)
			if step % outputFreq == 0:
				print("Training Step %d" % (step))
				print("Loss %f"          % (sess.run(loss)))
				#print("Weight %f"        % (sess.run(w)[0]))
				#print("Bias %f"          % (sess.run(b)[0]))
				self.Loss_train.write(str(step) + " " + str(sess.run(loss)) + "\n")
				self.hp_train.write(str(sess.run(w)) + " " + str(sess.run(b)) + "\n")

		#x = np.arange(-np.pi, np.pi + 2*np.pi/360, 2*np.pi/360)
		
		y = tf.cast(sess.run(w)*x_data, tf.float32) + tf.cast(sess.run(b), tf.float32)

		for i in range(len(self.cv)):
			self.force_result.write(str(self.cv[i]) + " " + str(sess.run(y)[i]) + "\n")

		self.Loss_train.close()
		self.hp_train.close()
		self.force_result.close() 
		sess.close()

	def saveData(self):

		# save training data using pickle
		pass

if __name__ == "__main__":
	
	s = trainingNN("analysis/loss_500k.dat", "analysis/hyperparam_500k.dat", "analysis/force_result_500k.dat")
	s.readData("trainingSet/wABFHistogram_test.dat", "trainingSet/wABF_Force_test.dat", "early_stage/wABF_Force001.dat")
	s.training(0.001, 5000001, 100)
#	s.readData("trainingSet/wABFHistogram004.dat", "trainingSet/wABF_Force004.dat")
#	s.readData("trainingSet/wABFHistogram005.dat", "trainingSet/wABF_Force005.dat")
