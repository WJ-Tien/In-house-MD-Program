#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np

class trainingANN(object):
	
	def __init__(self, fileLoss, filehyperparam, ndims, size):

		self.Loss_train     = open(str(fileLoss), "a")
		self.hp_train       = open(str(filehyperparam), "a")
		self.ndims          = ndims
		self.size           = size
		self.estForce       = np.zeros(self.size, dtype=np.float32)
		self.colvars        = np.linspace(-np.pi, np.pi, self.size*2)

	def addDenseLayer(self, input_neuron_size, output_neuron_size, activationFunc=None, nameFlag=None, *input_colvars):

		weights = []
		biases = tf.Variable(tf.truncated_normal([1, output_neuron_size], stddev=0.5))
		matmul_xW_add_bias = tf.zeros([self.ndims * self.size**self.ndims, output_neuron_size])

		for i in range(len(input_colvars)): # deal with nth colvar
			weights.append(tf.Variable(tf.truncated_normal([input_neuron_size, output_neuron_size], stddev=0.5)))
			matmul_xW_add_bias += tf.matmul(input_colvars[i], weights[i])
		matmul_xW_add_bias += biases 
		
		if nameFlag != None:
			matmul_xW_add_bias = tf.identity(matmul_xW_add_bias, name=nameFlag) 

		if activationFunc != None:
			return activationFunc(matmul_xW_add_bias), weights, biases 
		else:
			return matmul_xW_add_bias, weights, biases

	def training(self, array_colvar_to_train, array_target_to_learn, learningRate, regularFactor, epochs, outputFreq):

		self.hp_train.write("Regularization factor" + " " + str(regularFactor) + "\n")
		self.hp_train.write("Learning_rate"         + " " + str(learningRate) + "\n")
		self.hp_train.write("\n")
		self.hp_train.close()

		if self.ndims == 1:
			# 1-20-16-1
			CV                    = tf.placeholder(tf.float32, [None, 1], name="colvars")  # feature
			target                = tf.placeholder(tf.float32, [None, 1], name="targets")  # real data; training set; label	
			array_colvar_to_train = array_colvar_to_train[:, np.newaxis] # 361*1
			array_target_to_learn = array_target_to_learn[:, np.newaxis]

			layer1, w1, b1        = self.addDenseLayer(1, 20, tf.nn.sigmoid, None, CV)
			layer2, w2, b2        = self.addDenseLayer(20, 16, tf.nn.sigmoid, None, layer1)
			layerOutput, w3, b3   = self.addDenseLayer(16, 1, None, "annOutput", layer2)

			variables_to_feed     = {CV: array_colvar_to_train, target: array_target_to_learn}

		if self.ndims == 2:
			# 1-20-16-1
			CV_X, CV_Y            = np.meshgrid(array_colvar_to_train, array_colvar_to_train, indexing="ij") # 41 ---> 41*41
			CV_X                  = CV_X.reshape(CV_X.size) # 2D (41*41) --> 1D (1681)
			CV_X                  = np.append(CV_X, CV_X)[:, np.newaxis] #1681 * 1 --> 3362 * 1
			CV_Y                  = CV_Y.reshape(CV_Y.size)
			CV_Y                  = np.append(CV_Y, CV_Y)[:, np.newaxis] #1681 * 1 --> 3362 * 1

			CV_x                  = tf.placeholder(tf.float32, [self.size**self.ndims, 1], name="colvars_x")
			CV_y                  = tf.placeholder(tf.float32, [self.size**self.ndims, 1], name="colvars_y")
			target                = tf.placeholder(tf.float32, [self.size**self.ndims, 1], name="target")  

			array_target_to_learn = array_target_to_learn.reshape(self.ndims * self.size**self.ndims)[:, np.newaxis] # 3362 * 1

			layer1, w1, b1        = self.addDenseLayer(1, 20, tf.nn.sigmoid, None, CV_x, CV_y)
			layer2, w2, b2        = self.addDenseLayer(20, 16, tf.nn.sigmoid, None, layer1)
			layerOutput, w3, b3   = self.addDenseLayer(16, 1, None, "annOutput", layer2)
			variables_to_feed     = {CV_x: CV_X, CV_y: CV_Y, target: array_target_to_learn}

		loss         = tf.reduce_mean(tf.square(layerOutput - target) + regularFactor*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))*2) 
		optimizer    = tf.train.AdamOptimizer(learning_rate=learningRate) 
		train        = optimizer.minimize(loss)
		#optimizer    = tf.train.GradientDescentOptimizer(learningRate)
		#optimizer    = tf.train.AdagradOptimizer(learning_rate=learningRate) 

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):
				sess.run(train, feed_dict=variables_to_feed)
				if epoch % outputFreq == 0:
					self.Loss_train.write(str(epoch) + " " + str(sess.run(loss, feed_dict=variables_to_feed)) + "\n")
			self.Loss_train.write("\n")
			self.Loss_train.close()	
			
			self.estForce = (sess.run(layerOutput, feed_dict=variables_to_feed)).reshape(self.size) if self.ndims == 1 else \
                      (sess.run(layerOutput, feed_dict=variables_to_feed)).reshape(self.ndims, self.size, self.size)

		tf.reset_default_graph()

		return self.estForce

if __name__ == "__main__":
	pass
