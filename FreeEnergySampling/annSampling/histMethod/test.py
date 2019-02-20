#!/usr/bin/env python3
class test(object):
	def __init__(self):
		self.x = 1

	def func(self):
		self.gradient = 10
	def func2(self):
		y = self.gradient
		return y
if __name__ == "__main__":
	a = test()
	a.func()
	print(a.func2())
