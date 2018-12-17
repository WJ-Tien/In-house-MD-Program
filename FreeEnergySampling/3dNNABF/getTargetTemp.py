#!/usr/bin/env python3
import os
import sys

curdir = os.getcwd()

save = open("checkTargetTemp.dat", "w")
mass = float(sys.argv[1]) 

for filename in os.listdir(curdir):
	avg_vel_sq = 0.0 
	count      = 0
	#count_real = 0
	if filename.startswith("wABF_m"):
		with open(filename, "r") as fin:
			for line in fin:
				line = line.split()
				if line[0] != "#":
					count += 1
					avg_vel_sq += ((float(line[3]))**2)

		avg_vel_sq /= count
		avg_vel_sq *= mass
		save.write(str(filename) + "    " + str(avg_vel_sq) + "\n")


save.close()
					
