import numpy as np

x = np.random.normal(0,1,10000)
t = np.random.normal(0,1,10000)
acc = 0
x_sq = 0
for i in range(10000):
	acc += x[i]*t[i]
	x_sq += x[i]**2 
print(acc/10000)
print(x_sq/10000)
