#!/usr/bin/env python3
import numpy as np


#a = np.ones((1,361))
#b = np.full((1,361), 2)

#a = (a/b)
'''
acc = 0
acc_sq = 0
Racc =  0
def R(base1, base2):
	return (0.5) * base1 + 0.288675 * base2 
'''

acc = 0

for i in range(100000):
	random_xi = np.random.normal(0, 1)
	random_theta = np.random.normal(0, 1)
	acc += random_xi


print(acc/100000)
