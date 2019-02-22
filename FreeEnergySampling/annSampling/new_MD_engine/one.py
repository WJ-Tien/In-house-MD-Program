#!/usr/bin/env python3
from two import test


#def new_base():
#	return 7777 
#two.base = new_base

#from three import three
#print(three())

b = test(30)

def decorator(func):
	def _wrapper(num):
		return func(num) + b.a()
	return _wrapper

@decorator
def wrapped(num):
	return num 

print(wrapped(10))





	



