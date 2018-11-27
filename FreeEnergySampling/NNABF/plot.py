#!/usr/bin/env python3
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def RELU(x):
	return np.maximum(0,x)

w1 = 0.952158
b  = -0.193194 
#w2 = 1.99926877 
#w3 = 2.99907255
#w4 = 1.00055647
#w5 = 2.00015068 
#w6 = 3.00007939 

x = np.arange(-np.pi, np.pi + 2*np.pi/360, 2*np.pi/360)
#y = sigmoid(w1*x + b)
y = RELU(w1*x + b)
#y = w1 * np.sin(w4*x) + w2 * np.sin(w5*x) + w3 * np.sin(w6*x) + b 

with open("predicted.dat", "w") as p:
	for i in range(len(x)):
		p.write(str(x[i]) + " " + str(y[i]) + "\n")
