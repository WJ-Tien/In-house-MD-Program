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
noNN      = readData(sys.argv[3]) # Force_data: ABF without NN
noNNnoABF = readData(sys.argv[4]) # Force_data: without ABF and without NN 

avg_NN_error         = np.sqrt(np.sum(np.square(est - NN))        / len(est))
avg_noNN_error       = np.sqrt(np.sum(np.square(est - noNN))      / len(est))
avg_noNN_noABF_error = np.sqrt(np.sum(np.square(est - noNNnoABF)) / len(est))

print("avg_NN_error: "        , avg_NN_error)
print("avg_noNN_error: "      , avg_noNN_error)
print("avg_noNN_noABF_error: ", avg_noNN_noABF_error)

		
