#!/usr/bin/env python3
import numpy as np
binw = 2*np.pi/360
x = np.arange(-np.pi, np.pi + binw, binw)
y = np.sin(x) + 2*np.sin(2*x) + 3*np.sin(3*x)
print(max(y))
