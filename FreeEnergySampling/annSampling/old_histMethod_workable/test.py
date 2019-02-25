#!/usr/bin/env python3

class test(object):
	def __init__(self, x):
		self.x = x

	def run(self, x):
		self.x += x
		return self.x


a = test(1)
print(a.run(1))
		
