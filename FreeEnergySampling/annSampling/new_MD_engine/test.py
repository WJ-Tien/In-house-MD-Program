#!/usr/bin/env python3
import numpy as np

lin = 0
sq = 0
n = 100000
for i in range(n):
	r = np.random.uniform(-np.sqrt(3), np.sqrt(3))
	lin += r 
	sq += r**2
		

print(lin/n)
print(sq/n - (lin/n)**2)
#print(np.sqrt(sq/n - (lin/n)**2))
#print(sq/n)


