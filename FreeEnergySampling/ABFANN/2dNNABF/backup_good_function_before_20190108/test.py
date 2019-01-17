#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

bins = np.array([1,2,3,4,5])
c  = np.zeros(len(bins)+1)
inds = np.digitize(2.5, bins)
c[inds] += 1
print(c)

