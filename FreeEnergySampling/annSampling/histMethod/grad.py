#!/usr/bin/env python3
import tensorflow as tf

a = tf.constant(0.)
b = 2 * a *2
g = tf.gradients(a +b, [a, b])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(g))
