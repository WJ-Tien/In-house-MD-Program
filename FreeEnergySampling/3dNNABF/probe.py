#!/usr/bin/env python3

import numpy as np

class probe(object):
	
	def __init__(self, grid, rate):
		self.grid = grid
		self.rate = rate 
	
	def gradient(self):
		#TODO
		pass

	def localSearch(self, basis): # basis = array
		for i in basis:
			i -= rate * gradient

if __name__ == "__main__":
	s = probe(10, 0.01)
