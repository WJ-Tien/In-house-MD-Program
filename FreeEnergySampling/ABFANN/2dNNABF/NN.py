#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np

class trainingNN(object):
	
	def __init__(self, fileLoss, filehyperparam, fileweight, filebias, ndims, size):

		self.Loss_train     = open(str(fileLoss), "w")
		self.hp_train       = open(str(filehyperparam), "w")
		self.ndims          = ndims
		self.size           = size
		self.trainingLayers = 5 
		self.saveWeightArr  = np.zeros((self.trainingLayers, (self.size)**self.ndims), dtype=np.float32)
		self.saveBiasArr    = np.zeros((self.trainingLayers, (self.size)**self.ndims), dtype=np.float32)
		self.estForce       = np.zeros((self.size), dtype=np.float32)

	def training(self, array_weightList, array_biasList, array_colvar_to_train, array_force_to_learn, learning_rate, regularFactor, epochs, outputFreq):

		self.hp_train.write("Regularization factor" + " " + str(regularFactor) + "\n")
		self.hp_train.write("Learning_rate"         + " " + str(learning_rate) + "\n")
		self.hp_train.close()

		# assume 1 input layer; 4 hidden layers; 1 output layer; each with one neuron
		# activation function: ReLU 

		# data initialization
		if np.sum(array_weightList) == 0.0 and np.sum(array_biasList) == 0.0:
			if self.ndims == 1:			
				w1           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w2           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w3           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w4           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w5           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				b1           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-0.2, 0.2)), dtype=tf.float32)
				b2           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-0.4, 0.4)), dtype=tf.float32)
				b3           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-0.3, 0.3)), dtype=tf.float32)
				b4           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-0.5, 0.5)), dtype=tf.float32)
				b5           = tf.Variable(np.full((len(array_force_to_learn)), np.random.uniform(-0.1, 0.1)), dtype=tf.float32)

			if self.ndims == 2:			
				w1           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w2           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w3           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w4           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				w5           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
				b1           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-0.2, 0.2)), dtype=tf.float32)
				b2           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-0.4, 0.4)), dtype=tf.float32)
				b3           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-0.3, 0.3)), dtype=tf.float32)
				b4           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-0.5, 0.5)), dtype=tf.float32)
				b5           = tf.Variable(np.full((len(array_force_to_learn), len(array_force_to_learn)), np.random.uniform(-0.1, 0.1)), dtype=tf.float32)

		# data received from previous training routine
		else:
			w1 = tf.Variable(array_weightList[0], dtype=tf.float32)
			w2 = tf.Variable(array_weightList[1], dtype=tf.float32)
			w3 = tf.Variable(array_weightList[2], dtype=tf.float32)
			w4 = tf.Variable(array_weightList[3], dtype=tf.float32)
			w5 = tf.Variable(array_weightList[4], dtype=tf.float32)
			b1 = tf.Variable(array_biasList[0], dtype=tf.float32)
			b2 = tf.Variable(array_biasList[1], dtype=tf.float32)
			b3 = tf.Variable(array_biasList[2], dtype=tf.float32)
			b4 = tf.Variable(array_biasList[3], dtype=tf.float32)
			b5 = tf.Variable(array_biasList[4], dtype=tf.float32)
	
		x           = tf.placeholder(tf.float32, [len(array_colvar_to_train)])  # feature
		y           = tf.placeholder(tf.float32, [len(array_force_to_learn)])  # real data; training set; label

		y_estimated = tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(w1 * x + b1) * w2 + b2)) * w3 + b3) * w4 + b4) * w5 + b5

		regularizer = tf.nn.l2_loss(w1 + w2 + w3 + w4 + w5) * 2
		loss        = tf.reduce_mean(tf.square(y_estimated - y))
		loss        = tf.reduce_mean(loss + regularFactor * regularizer)
		optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
		train       = optimizer.minimize(loss)

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):
				sess.run(train, feed_dict={x: array_colvar_to_train, y: array_force_to_learn})

				if epoch % outputFreq == 0:
					self.Loss_train.write(str(epoch) + " " + str(sess.run(loss, feed_dict={x: array_colvar_to_train, y: array_force_to_learn})) + "\n")
			self.Loss_train.close()	

			y_estimated           = sess.run(y_estimated, feed_dict={x: array_colvar_to_train})
			self.saveWeightArr[0] = sess.run(w1)
			self.saveWeightArr[1] = sess.run(w2)
			self.saveWeightArr[2] = sess.run(w3)
			self.saveWeightArr[3] = sess.run(w4)
			self.saveWeightArr[4] = sess.run(w5)
			self.saveBiasArr[0]   = sess.run(b1)
			self.saveBiasArr[1]   = sess.run(b2)
			self.saveBiasArr[2]   = sess.run(b3)
			self.saveBiasArr[3]   = sess.run(b4)
			self.saveBiasArr[4]   = sess.run(b5)
		


		return self.saveWeightArr, self.saveBiasArr, y_estimated 

if __name__ == "__main__":

	pass	
	#import pickle
	#self.save_weight     = open(str(fileweight), "wb")	
	#self.save_bias       = open(str(filebias), "wb")	
	#pickle.dump(list(sess.run(w1)),  self.save_weight)
	#pickle.dump(list(sess.run(b1)),  self.save_bias)
