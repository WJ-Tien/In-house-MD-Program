#!/usr/bin/env python3
import numpy as np

a = np.linspace(-3, 3, 40)
#print(a)
#print(np.digitize(-3.001, a))
#print(np.digitize(-3, a))
print(a)
print(np.digitize(2.99, a))
print(np.digitize(3, a))
print(np.digitize(-3, a))
print(np.digitize(3.01, a))


