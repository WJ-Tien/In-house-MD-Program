#!/usr/bin/env python3
import os
import sys

curdir = os.getcwd()

save = open("checkTargetTemp.dat", "w")
mass = float(sys.argv[1]) 
ndims = int(sys.argv[2])

for filename in os.listdir(curdir):
	avg_vel_sq = 0.0 
	count      = 0
	if filename.startswith("conventional"):
		with open(filename, "r") as fin:
			if ndims == 1:
				for line in fin:
					line = line.split()
					if line[0] != "#":
						count += 1
						avg_vel_sq += ((float(line[3]))**2)

				avg_vel_sq /= (count)
				TargetTemp = (avg_vel_sq * (mass / ndims))
				save.write(str(filename) + "    " + str(TargetTemp) + "\n")

			if ndims == 2:
				for line in fin:
					line = line.split()
					if line[0] != "#":
						count += 1
						avg_vel_sq += ((float(line[3]))**2)
						avg_vel_sq += ((float(line[5]))**2)

				avg_vel_sq /= (count)
				TargetTemp = (avg_vel_sq * (mass/ndims))
				save.write(str(filename) + "    " + str(TargetTemp) + "\n")

save.close()
					
