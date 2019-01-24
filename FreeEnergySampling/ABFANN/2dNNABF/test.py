#!/usr/bin/env python3
import numpy as np

#x = np.linspace(-3, 3, 41) 
#y = np.linspace(-3, 3, 41)
#X, Y = np.meshgrid(x, y, indexing="ij")
#print(X.size)
#print(X.reshape(X.size))
#print(Y.reshape(Y.size))
#a = np.array([[[1,2], [3,4],[5,6]], [[11,22], [33,44],[15,16]]])
#print(a.shape)
a = [1,2,3]
b = [1,2]
#a.append(b)
a[2] = b
print(a)
