#!/usr/bin/env python3

import sys
gamma = 0.05
mass  = float(sys.argv[2])
dragForce = 0
count = 0
with open(sys.argv[1], "r") as fin:
	for line in fin:
		if line[0] != "#":
			dragForce += -mass*gamma*float(line.split()[3])
			count += 1


print(dragForce/count)


			
			
