#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np
#import pickle

class trainingNN(object):
	
	def __init__(self, fileLoss, filehyperparam, fileweight, filebias, ndims):


		self.Loss_train      = open(str(fileLoss), "w")
		self.hp_train        = open(str(filehyperparam), "w")
		self.saveWeightArr   = np.zeros((5, 361), dtype=np.float32)
		self.saveBiasArr     = np.zeros((5, 361), dtype=np.float32)
		self.ndims           = ndims

		#self.save_weight     = open(str(fileweight), "wb")	
		#self.save_bias       = open(str(filebias), "wb")	

	def training(self, array_weightList, array_biasList, array_force_to_train, array_force_to_learn, learning_rate, regularFactor, epoch, outputFreq):

		# assume 1 input layer; 4 hidden layers; 1 output layer; each with one neuron
		# activation function: relu 

		# data initialization
		x_data       = tf.cast(np.array(array_force_to_train), tf.float32)

		if array_weightList.size == 0 and array_biasList.size == 0:
			w1           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
			w2           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
			w3           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
			w4           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
			w5           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-1, 1)),     dtype=tf.float32)
			b1           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-0.2, 0.2)), dtype=tf.float32)
			b2           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-0.4, 0.4)), dtype=tf.float32)
			b3           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-0.3, 0.3)), dtype=tf.float32)
			b4           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-0.5, 0.5)), dtype=tf.float32)
			b5           = tf.Variable(np.full((self.ndims, len(array_force_to_learn)), np.random.uniform(-0.1, 0.1)), dtype=tf.float32)

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
	
		y_real      = np.array(array_force_to_train)  
		y_estimated = tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(w1 * x_data + b1) * w2 + b2)) * w3 + b3) * w4 + b4) * w5 + b5

		# define loss function and associated optimizer
		regularizer = tf.nn.l2_loss(w1 + w2 + w3 + w4 + w5) * 2
		loss        = tf.reduce_mean(tf.square(y_estimated - y_real))
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
		self.hp_train.write("Learning_rate"         + " " + str(learning_rate) + "\n")

		# After training
		y = tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(tf.nn.relu(sess.run(w1) * x_data + sess.run(b1)) * sess.run(w2) + sess.run(b2))) * sess.run(w3) + sess.run(b3)) * sess.run(w4) + sess.run(b4)) *sess.run(w5) + sess.run(b5)

		
		# save weights and biases via pickle
		#pickle.dump(list(sess.run(w1)),  self.save_weight)
		#pickle.dump(list(sess.run(b1)),  self.save_bias)
		#pickle.dump(list(sess.run(w2)), self.save_weight)
		#pickle.dump(list(sess.run(b2)), self.save_bias)
		#pickle.dump(list(sess.run(w3)), self.save_weight)
		#pickle.dump(list(sess.run(b3)), self.save_bias)
		#pickle.dump(list(sess.run(w4)), self.save_weight)
		#pickle.dump(list(sess.run(b4)), self.save_bias)

		est_force             = list(sess.run(y))
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

		#self.save_weight.close()
		#self.save_bias.close()
		self.Loss_train.close()
		self.hp_train.close()
		sess.close()
	
		return self.saveWeightArr, self.saveBiasArr, est_force 

if __name__ == "__main__":
	pass	
