#!/usr/bin/env python3
import numpy as np

random_acc1 = 0
random_acc2 = 0
TM1 = np.sqrt(2*0.05*1*0.01/0.005)
TM2 = 0.00001 

for i in range(10000000):
	random_acc1 += np.sqrt(TM1)*np.random.normal(0, 1)
	#random_acc2 += np.sqrt(TM2)*np.random.normal(0, 1)


print(random_acc1/10000000)
#print(random_acc2/10000)
