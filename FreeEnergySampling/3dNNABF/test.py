#!/usr/bin/env python3
import numpy as np


#a = np.ones((1,361))
#b = np.full((1,361), 2)

#a = (a/b)
acc = 0
acc_sq = 0
for i in range(10000):
	base = np.random.normal(0, 1)
	acc += base
	acc_sq +=  (base**2)
print(acc/10000)
print(acc_sq/10000)

