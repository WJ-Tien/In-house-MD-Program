#!/usr/bin/env python3
import numpy as np

"""
acc = 0
n = 1000000
sq = 0
for i in range(n):
	a = np.random.uniform(-np.sqrt(3), np.sqrt(3))
	acc += a 
	sq += a**2


#print((b-a)**2/ 12)
print("average --> %f" %(acc/n))
print("sq_average --> %f" %(sq/n))
"""

a = np.ones((2,10,10))
print(np.sum(a))

