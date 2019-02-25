#!/usr/bin/env python3

import numpy as np
import sys

def readData(f):
	with open(f, "r") as fin:
		saveForce = []
		for line in fin:
			saveForce.append(float(line.split()[1]))
	return np.array(saveForce)

est       = readData(sys.argv[1]) # estimate
NN        = readData(sys.argv[2]) # Force_data: ABF with NN

avg_NN_error         = np.sqrt(np.sum(np.square(est - NN))        / len(est))

print("avg_error: "        , avg_NN_error)



		
