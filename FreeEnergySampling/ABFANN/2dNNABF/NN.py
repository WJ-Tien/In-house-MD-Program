#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np

class trainingNN(object):
	
	def __init__(self, fileLoss, filehyperparam, ndims, size):

		self.Loss_train     = open(str(fileLoss), "a")
		self.hp_train       = open(str(filehyperparam), "w")
		self.ndims          = ndims
		self.size           = size
		self.trainingLayers = 5 
		self.saveWeightArr  = np.zeros((self.trainingLayers, (self.size)**self.ndims), dtype=np.float32)
		self.saveBiasArr    = np.zeros((self.trainingLayers, (self.size)**self.ndims), dtype=np.float32)
		self.estForce       = np.zeros(self.size, dtype=np.float32)

	#def training(self, array_weightList, array_biasList, array_colvar_to_train, array_force_to_learn, learning_rate, regularFactor, epochs, outputFreq):
	def training(self, array_colvar_to_train, array_force_to_learn, learning_rate, regularFactor, epochs, outputFreq):

		self.hp_train.write("Regularization factor" + " " + str(regularFactor) + "\n")
		self.hp_train.write("Learning_rate"         + " " + str(learning_rate) + "\n")
		self.hp_train.close()

		# assume 1 input layer; 4 hidden layers; 1 output layer; each with one neuron
		# activation function: ReLU 

		# data initialization
		#if np.sum(array_weightList) == 0.0 and np.sum(array_biasList) == 0.0:
		array_colvar_to_train = array_colvar_to_train[:, np.newaxis] # 361*1
		array_force_to_learn  = array_force_to_learn[:, np.newaxis]

		x  = tf.placeholder(tf.float32, [len(array_colvar_to_train), 1])  # feature
		y  = tf.placeholder(tf.float32, [len(array_force_to_learn), 1])  # real data; training set; label	

		# layer 1
		node_12     = 48		
		w1          = tf.Variable(tf.truncated_normal([1, node_12], stddev=0.1)) #361*1 * 1*4 = 361*4
		b1          = tf.Variable(tf.zeros([self.size, node_12]))
		y1          = tf.nn.relu(tf.matmul(x, w1) + b1)
			
		# layer 2
		node_23     = 24	
		w2          = tf.Variable(tf.truncated_normal([node_12, node_23], stddev=0.1)) #361*4 * (4*2)=361*2
		b2          = tf.Variable(tf.zeros([self.size, node_23]))
		y2          = tf.nn.relu(tf.matmul(y1, w2) + b2)

		node_34     = 6		
		w3          = tf.Variable(tf.truncated_normal([node_23, node_34], stddev=0.1)) #361*4 * (4*2)=361*2
		b3          = tf.Variable(tf.zeros([self.size, node_34]))
		y3          = tf.nn.relu(tf.matmul(y2, w3) + b3)

		# layer 3
		node_45     = 1		
		w4          = tf.Variable(tf.truncated_normal([node_34, node_45], stddev=0.1)) #361*2* 2*1 = 361*1
		b4          = tf.Variable(tf.zeros([self.size, 1]))
		y_estimated = (tf.matmul(y3, w4) + b4)

		# data received from previous training routine

		#regularizer = tf.nn.l2_loss(w1.flatten()+w2.flatten() + w3.flatten()) * 2
		loss        = tf.reduce_mean(tf.square(y_estimated - y))
		#loss        = tf.reduce_mean(loss + regularFactor * regularizer)
		optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
		train       = optimizer.minimize(loss)

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):
				sess.run(train, feed_dict={x: array_colvar_to_train, y: array_force_to_learn})

				if epoch % outputFreq == 0:
					self.Loss_train.write(str(epoch) + " " + str(sess.run(loss, feed_dict={x: array_colvar_to_train, y: array_force_to_learn})) + "\n")

			self.Loss_train.close()	
			
			self.estForce          = sess.run(y_estimated, feed_dict={x: array_colvar_to_train, y:array_force_to_learn})
			#self.saveWeightArr[0] = sess.run(w1) 
			#self.saveWeightArr[1] = sess.run(w2)
			#self.saveWeightArr[2] = sess.run(w3)
			#self.saveWeightArr[3] = sess.run(w4)
			#self.saveWeightArr[4] = sess.run(b5)
			#self.saveBiasArr[0]   = sess.run(b1)
			#self.saveBiasArr[1]   = sess.run(b2)
			#self.saveBiasArr[2]   = sess.run(b3)
			#self.saveBiasArr[3]   = sess.run(b4)
			#self.saveBiasArr[4]   = sess.run(b5)

		tf.reset_default_graph()

		#return self.saveWeightArr, self.saveBiasArr, self.estForce 
		return self.estForce

if __name__ == "__main__":
	
	output = trainingNN("loss.dat", "hyperparam.dat", 1, 361)
	array_force_to_learn = []

	with open("estimate", "r") as fin:
		for line in fin:
			array_force_to_learn.append(line.split()[1])
	array_force_to_learn = np.array(array_force_to_learn)
	array_colvar_to_train = np.linspace(-np.pi, np.pi, 361)

	#force = output.training(array_colvar_to_train, array_force_to_learn, learning_rate=0.3, regularFactor=0, epochs=5000, outputFreq=100)
	force = output.training(array_colvar_to_train, array_force_to_learn, learning_rate=0.005, regularFactor=0, epochs=5000, outputFreq=100)
	with open("out", "w") as fout:
		for cv, force in zip(array_colvar_to_train, force):
			fout.write(str(cv) +  " " + str(force[0]) + "\n")
	
	#import pickle
	#self.save_weight     = open(str(fileweight), "wb")	
	#self.save_bias       = open(str(filebias), "wb")	
	#pickle.dump(list(sess.run(w1)),  self.save_weight)
	#pickle.dump(list(sess.run(b1)),  self.save_bias)
