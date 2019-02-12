#!/usr/bin/env python3
import tensorflow as tf

#a = tf.constant(0.)
a = tf.placeholder(dtype=tf.float32, shape=(1))
b = 2 * a
g = tf.gradients(a + b, [a, b])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(g, feed_dict={a: 1.0}))
