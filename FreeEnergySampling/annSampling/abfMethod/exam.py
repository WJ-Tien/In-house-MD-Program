#!/usr/bin/env python3
import numpy as np

class first(object):
	def __init__(self, ndims, x):
		self.ndims = ndims 
		self.x     = x

	def test(self):
		return 1

class second(first):
	def __init__(self, ndims, x,  out):
		super().__init__(ndims, x)
		self.out = out + 10000

if __name__ == "__main__":
	pass
	


	

	
