#!/usr/bin/env python3

import tensorflow as tf 
import numpy as np

class trainingANN(object):
  
  def __init__(self, fileLoss, filehyperparam, ndims, size):

    self.Loss_train = open(str(fileLoss), "a")
    self.hp_train   = open(str(filehyperparam), "a")
    self.ndims      = ndims
    self.size       = size
    self.estTarget  = np.zeros((self.size)) if self.ndims == 1 else np.zeros((self.ndims, self.size, self.size))  

  def addDenseLayer(self, input_neuron_size, output_neuron_size, activationFunc=None, nameFlag=None, *input_colvars):

    weights = []
    biases = tf.Variable(tf.truncated_normal([1, output_neuron_size], stddev=0.5))

    matmul_xW_add_bias = tf.zeros([tf.shape(input_colvars[0])[0], output_neuron_size])

    for i in range(len(input_colvars)): # loop over n colvars
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
      CV                    = tf.placeholder(tf.float32, [None, 1], name="colvars") # feature
      array_colvar_to_train = array_colvar_to_train[:, np.newaxis]                  # 361*1

      target                = tf.placeholder(tf.float32, [None, 1], name="targets") # real data; training set; label  
      array_target_to_learn = array_target_to_learn[:, np.newaxis]

      layer1, w1, b1        = self.addDenseLayer(1, 96, tf.nn.sigmoid, None, CV)
      layer2, w2, b2        = self.addDenseLayer(96, 72, tf.nn.sigmoid, None, layer1)
      layer3, w3, b3        = self.addDenseLayer(72, 36, tf.nn.sigmoid, None, layer2)
      layerOutput, w4, b4   = self.addDenseLayer(36, 1, None, "annOutput", layer3)
      variables_to_feed     = {CV: array_colvar_to_train, target: array_target_to_learn}
      loss                  = tf.reduce_mean(tf.square(layerOutput - target) + regularFactor*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4))*2) 

    if self.ndims == 2: 
      # 1(2)-30-24-2
      CV_x                  = tf.placeholder(tf.float32, [None, 1], name="colvars_x")
      CV_y                  = tf.placeholder(tf.float32, [None, 1], name="colvars_y")
      CV_X, CV_Y            = np.meshgrid(array_colvar_to_train, array_colvar_to_train, indexing="ij") # 41 ---> 41*41
      CV_X                  = CV_X.reshape(CV_X.size)[:, np.newaxis]                                   # 2D (41*41) --> 2D (1681*1)
      CV_Y                  = CV_Y.reshape(CV_Y.size)[:, np.newaxis]

      target_x              = tf.placeholder(tf.float32, [None, 1], name="target_x")  
      target_y              = tf.placeholder(tf.float32, [None, 1], name="target_y")  
      array_target_x_to_learn = array_target_to_learn[0].reshape(array_target_to_learn[0].shape[0] * array_target_to_learn[0].shape[1])[:, np.newaxis] # 1681 * 1
      array_target_y_to_learn = array_target_to_learn[1].reshape(array_target_to_learn[1].shape[0] * array_target_to_learn[1].shape[1])[:, np.newaxis] # 1681 * 1


      layer1, w1, b1        = self.addDenseLayer(1, 30, tf.nn.sigmoid, None, CV_x, CV_y)
      layer2, w2, b2        = self.addDenseLayer(30, 24, tf.nn.sigmoid, None, layer1)
      layerOutput, w3, b3   = self.addDenseLayer(24, 2, None, "annOutput", layer2) #1681*2
      layerOutput_x         = layerOutput[:, 0][:, np.newaxis] # 1681 * 1
      layerOutput_y         = layerOutput[:, 1][:, np.newaxis] # 1681 * 1
      variables_to_feed     = {CV_x: CV_X, CV_y: CV_Y, target_x: array_target_x_to_learn, target_y: array_target_y_to_learn}
      loss                  = tf.reduce_mean(tf.square(layerOutput_x - target_x) + tf.square(layerOutput_y - target_y) + regularFactor*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))*2) 

    if self.ndims == 3: 
      pass

    # https://stackoverflow.com/questions/49953379/tensorflow-multiple-loss-functions-vs-multiple-training-ops
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate) 
    train     = optimizer.minimize(loss)

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())

      for epoch in range(epochs):
        sess.run(train, feed_dict=variables_to_feed)
        if epoch % outputFreq == 0:
          self.Loss_train.write(str(epoch) + " " + str(sess.run(loss, feed_dict=variables_to_feed)) + "\n")
      self.Loss_train.write("\n")
      self.Loss_train.close() 

      # TODO better structure   
      reshape_estTarget = sess.run(layerOutput, feed_dict=variables_to_feed) # remove the np.newaxis

      if self.ndims == 1:
        self.estTarget = reshape_estTarget.reshape(self.size)

      if self.ndims == 2:
        estTargetX = reshape_estTarget[:, 0].reshape(self.size, self.size)
        estTargetY = reshape_estTarget[:, 1].reshape(self.size, self.size)
        self.estTarget[0] = estTargetX
        self.estTarget[1] = estTargetY

      tf.train.Saver().save(sess, "net" + str(self.ndims) + "D" + "/netSaver.ckpt")

    tf.reset_default_graph()

    return self.estTarget

if __name__ == "__main__":
  pass
