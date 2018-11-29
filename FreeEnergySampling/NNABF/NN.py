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
		
	def readData(self, file_force_to_train, file_force_to_learn): 
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
	
	def training(self, learning_rate, step, outputFreq):

		# assume 1 input layer; 4 hidden layer; 1 output layer; each with one neuron
		# activation function: tanh and relu 

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
		
		#save_weight_l = open("weight.pkl", "rb")	
		#save_bias_l   = open("bias.pkl", "rb")	
		#w = tf.Variable(pickle.load(save_weight_l))
		#b = tf.Variable(pickle.load(save_bias_l))
		#w2 = tf.Variable(pickle.load(save_weight_l))
		#b2 = tf.Variable(pickle.load(save_bias_l))

		y_real      = np.array(self.force_to_train)  
		#y_predicted = tf.nn.tanh(tf.nn.tanh(tf.nn.relu(w * x_data + b) * w2 + b2)) * w3 + b3 #perfect!!!
		y_predicted = tf.nn.relu(tf.nn.tanh(tf.nn.tanh(tf.nn.relu(w * x_data + b) * w2 + b2)) * w3 + b3) * w4 + b4

		# define loss function and associated optimizer

		regularizer = tf.nn.l2_loss(w) * 2
		alpha       = 5 
		loss        = tf.reduce_mean(tf.square(y_predicted - y_real))
		loss        = tf.reduce_mean(loss + alpha * regularizer)
		optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
		train       = optimizer.minimize(loss)

		# initialize the "tensorflow graph"
		init = tf.global_variables_initializer()

		# run the graph (session)
		sess = tf.Session()
		sess.run(init)

		for steps in range(step):
			sess.run(train)
			if steps % outputFreq == 0:
				print("Training Step %d" % (steps))
				print("Loss %f"          % (sess.run(loss)))
				self.Loss_train.write(str(steps) + " " + str(sess.run(loss)) + "\n")

		self.hp_train.write("Regularization factor" + " " + str(alpha) + "\n")
		self.hp_train.write("learning_rate" + " " + str(learning_rate) + "\n")

		#y = tf.cast(sess.run(w)*x_data, tf.float32) + tf.cast(sess.run(b), tf.float32)
		y = sess.run(w4)*(sess.run(w3) * (sess.run(w2)*(sess.run(w)*x_data + sess.run(b)) + sess.run(b2)) + sess.run(b3)) + sess.run(b4)

		for i in range(len(self.cv)):
			self.force_result.write(str(self.cv[i]) + " " + str(0) + " " + str(0) + " " + str(sess.run(y)[i]) + "\n")

		save_weight = open("weight.pkl", "wb")	
		save_bias   = open("bias.pkl", "wb")	
		pickle.dump(list(sess.run(w)), save_weight)
		pickle.dump(list(sess.run(b)), save_bias)
		pickle.dump(list(sess.run(w2)), save_weight)
		pickle.dump(list(sess.run(b2)), save_bias)
		pickle.dump(list(sess.run(w3)), save_weight)
		pickle.dump(list(sess.run(b3)), save_bias)
		pickle.dump(list(sess.run(w4)), save_weight)
		pickle.dump(list(sess.run(b4)), save_bias)

		save_weight.close()
		save_bias.close()
		#save_weight_l.close()
		#save_bias_l.close()
		self.Loss_train.close()
		self.hp_train.close()
		self.force_result.close() 
		sess.close()


if __name__ == "__main__":
	
	s = trainingNN("analysis/loss_X.dat", "analysis/hyperparam_X.dat", "analysis/force_result_X.dat")
	s.readData("trainingSet/wABF_Force_test.dat", "early_stage/wABF_Force001.dat")
	#s.readData("trainingSet/wABF_Force_test.dat", "analysis/force_result_X.dat") #second time
	s.training(0.05, 50001, 100)
