#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np
#import pickle

class trainingNN(object):
	
	def __init__(self, fileLoss, filehyperparam, fileforceResult, fileweight, filebias):

		self.force_to_train  = []
		self.force_to_learn  = []
		self.cv              = []
		self.Loss_train      = open(str(fileLoss), "w")
		self.hp_train        = open(str(filehyperparam), "w")
		self.force_result    = open(str(fileforceResult), "w")
		#self.save_weight     = open(str(fileweight), "wb")	
		#self.save_bias       = open(str(filebias), "wb")	
	'''	
	def readData(self, file_force_to_train, file_force_to_learn): 
		self.fileIn_force_to_train = open(str(file_force_to_train), "r") # 0 3
		self.fileIn_force_to_learn = open(str(file_force_to_learn), "r") # 0 3

		for line in self.fileIn_force_to_train:
			self.cv.append(float(line.split()[0]))
			self.force_to_train.append(float(line.split()[3]))
		
		for line in self.fileIn_force_to_learn:
			#self.force_to_learn.append(float(line.split()[3]))

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
	'''
	def readData(self, file_force_to_train, array_force_to_learn): 
		self.fileIn_force_to_train = open(str(file_force_to_train), "r") # 0 3

		for line in self.fileIn_force_to_train:
			self.cv.append(float(line.split()[0]))
			self.force_to_train.append(float(line.split()[3]))

		self.fileIn_force_to_train.close()

		self.force_to_learn = array_force_to_learn

		if len(self.force_to_train) == len(self.force_to_learn):
			pass
		else:
			while(len(self.force_to_train) > len(self.force_to_learn)):
				self.force_to_train.pop() 
				self.cv.pop() 
			while(len(self.force_to_train) < len(self.force_to_learn)):
				self.force_to_learn.pop()
	
	def training(self, learning_rate, regularFactor, epoch, outputFreq):

		# assume 1 input layer; 4 hidden layers; 1 output layer; each with one neuron
		# activation function: relu 

		# data initialization
		self.cv     = np.array(self.cv)
		x_data      = tf.cast(np.array(self.force_to_learn), tf.float32)
		w           = tf.Variable(tf.random_uniform([len(self.force_to_train)], -1, 1), dtype=tf.float32)
		w2          = tf.Variable(tf.random_uniform([len(self.force_to_train)], -1, 1), dtype=tf.float32)
		w3          = tf.Variable(tf.random_uniform([len(self.force_to_train)], -1, 1), dtype=tf.float32)
		w4          = tf.Variable(tf.random_uniform([len(self.force_to_train)], -1, 1), dtype=tf.float32)
		b           = tf.Variable(tf.zeros([len(self.force_to_train)]), dtype=tf.float32)
		b2          = tf.Variable(tf.zeros([len(self.force_to_train)]), dtype=tf.float32)
		b3          = tf.Variable(tf.zeros([len(self.force_to_train)]), dtype=tf.float32)
		b4          = tf.Variable(tf.zeros([len(self.force_to_train)]), dtype=tf.float32)
		
		y_real      = np.array(self.force_to_train)  
		y_predicted = tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(w * x_data + b) * w2 + b2)) * w3 + b3) * w4 + b4

		# define loss function and associated optimizer
		regularizer = tf.nn.l2_loss(w) * 2
		loss        = tf.reduce_mean(tf.square(y_predicted - y_real))
		loss        = tf.reduce_mean(loss + regularFactor * regularizer)
		optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
		train       = optimizer.minimize(loss)

		# initialize the "tensorflow graph"
		init = tf.global_variables_initializer()

		# run the graph (session)
		sess = tf.Session()
		sess.run(init)

		for steps in range(epoch):
			sess.run(train)
			if steps % outputFreq == 0:
				self.Loss_train.write(str(steps) + " " + str(sess.run(loss)) + "\n")

		self.hp_train.write("Regularization factor" + " " + str(regularFactor) + "\n")
		self.hp_train.write("learning_rate"         + " " + str(learning_rate) + "\n")

		# After training
		y = sess.run(w4)*(sess.run(w3) * (sess.run(w2)*(sess.run(w)*x_data + sess.run(b)) + sess.run(b2)) + sess.run(b3)) + sess.run(b4)

		for i in range(len(self.cv)):
			self.force_result.write(str(self.cv[i]) + " " + str(0) + " " + str(0) + " " + str(sess.run(y)[i]) + "\n")
		
		# save weights and biases via pickle
		#pickle.dump(list(sess.run(w)),  self.save_weight)
		#pickle.dump(list(sess.run(b)),  self.save_bias)
		#pickle.dump(list(sess.run(w2)), self.save_weight)
		#pickle.dump(list(sess.run(b2)), self.save_bias)
		#pickle.dump(list(sess.run(w3)), self.save_weight)
		#pickle.dump(list(sess.run(b3)), self.save_bias)
		#pickle.dump(list(sess.run(w4)), self.save_weight)
		#pickle.dump(list(sess.run(b4)), self.save_bias)

		absol_loss = float(sess.run(loss))
		absol_force_result = list(sess.run(y))

		#self.save_weight.close()
		#self.save_bias.close()
		self.Loss_train.close()
		self.hp_train.close()
		self.force_result.close() 
		sess.close()
	
		return absol_force_result, absol_loss 


if __name__ == "__main__":
	pass	
	#s = trainingNN("analysis/loss.dat", "analysis/hyperparam.dat", "analysis/force_result.dat", "pklsave/weight.pkl", "pklsave/bias.pkl")
	#s.readData("trainingSet/wABF_Force_test.dat", "early_stage/wABF_Force001.dat")
	#s.training(0.05, 5, 50001, 100)
