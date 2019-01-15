#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np

class trainingNN(object):
	
	def __init__(self, fileLoss, filehyperparam, ndims, size):

		self.Loss_train     = open(str(fileLoss), "a")
		self.hp_train       = open(str(filehyperparam), "w")
		self.ndims          = ndims
		self.size           = size
		self.estForce       = np.zeros(self.size, dtype=np.float32)

	def training(self, array_colvar_to_train, array_force_to_learn, learning_rate, regularFactor, epochs, outputFreq):

		self.hp_train.write("Regularization factor" + " " + str(regularFactor) + "\n")
		self.hp_train.write("Learning_rate"         + " " + str(learning_rate) + "\n")
		self.hp_train.close()

		array_colvar_to_train = array_colvar_to_train[:, np.newaxis] # 361*1
		array_force_to_learn  = array_force_to_learn[:, np.newaxis]

		x  = tf.placeholder(tf.float32, [len(array_colvar_to_train), 1])  # feature
		y  = tf.placeholder(tf.float32, [len(array_force_to_learn), 1])  # real data; training set; label	

		# 1-64-48-1

		# layer 1
		node_12     = 96		
		w1          = tf.Variable(tf.truncated_normal([1, node_12], stddev=0.05)) #361*1 * 1*4 = 361*4
		b1          = tf.Variable(tf.zeros([self.size, node_12]))
		y1          = tf.nn.relu(tf.matmul(x, w1) + b1)
			
		# layer 2
		node_23     = 48	
		w2          = tf.Variable(tf.truncated_normal([node_12, node_23], stddev=0.05)) #361*4 * (4*2)=361*2
		b2          = tf.Variable(tf.zeros([self.size, node_23]))
		y2          = tf.nn.relu(tf.matmul(y1, w2) + b2)


		node_34     = 1 		
		w3          = tf.Variable(tf.truncated_normal([node_23, node_34], stddev=0.05)) #361*4 * (4*2)=361*2
		b3          = tf.Variable(tf.zeros([self.size, node_34]))
		y_estimated = (tf.matmul(y2, w3) + b3)

		# layer 3
		#node_45     = 1		
		#w4          = tf.Variable(tf.truncated_normal([node_34, node_45], stddev=0.05)) #361*2* 2*1 = 361*1
		#b4          = tf.Variable(tf.zeros([self.size, 1]))
		#y_estimated = (tf.matmul(y3, w4) + b4)


		#reg_losses   = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss         = tf.reduce_mean(tf.square(y_estimated - y) + regularFactor*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))*2) #+ tf.nn.l2_loss(w4))*2)
		#loss        += regularFactor * tf.reduce_sum(reg_losses)
		optimizer    = tf.train.GradientDescentOptimizer(learning_rate)
		train        = optimizer.minimize(loss)

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):
				sess.run(train, feed_dict={x: array_colvar_to_train, y: array_force_to_learn})

				if epoch % outputFreq == 0:
					self.Loss_train.write(str(epoch) + " " + str(sess.run(loss, feed_dict={x: array_colvar_to_train, y: array_force_to_learn})) + "\n")
			self.Loss_train.write("\n")

			self.Loss_train.close()	
			
			self.estForce          = sess.run(y_estimated, feed_dict={x: array_colvar_to_train, y:array_force_to_learn})
			self.estForce          = self.estForce.reshape(self.size)	

		tf.reset_default_graph()

		return self.estForce

if __name__ == "__main__":

	output = trainingNN("loss.dat", "hyperparam.dat", 1, 361)
	array_force_to_learn = []

	with open("estimate", "r") as fin:
		for line in fin:
			array_force_to_learn.append(line.split()[1])
	array_force_to_learn = np.array(array_force_to_learn)
	array_colvar_to_train = np.linspace(-np.pi, np.pi, 361)

	force = output.training(array_colvar_to_train, array_force_to_learn, learning_rate=0.024, regularFactor=0.0, epochs=2350, outputFreq=100)
	with open("out", "w") as fout:
		for cv, force in zip(array_colvar_to_train, force):
			fout.write(str(cv) +  " " + str(force) + "\n")
	
	
	#import pickle
	#self.save_weight     = open(str(fileweight), "wb")	
	#self.save_bias       = open(str(filebias), "wb")	
	#pickle.dump(list(sess.run(w1)),  self.save_weight)
	#pickle.dump(list(sess.run(b1)),  self.save_bias)
