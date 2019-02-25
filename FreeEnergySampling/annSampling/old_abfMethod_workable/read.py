#!/usr/bin/env python3
import sys
acc = 0
with open(sys.argv[1], "r") as fin:
	for line in fin:
		line = line.split()
		print(float(line[5]))
		acc += float(line[5])


print(acc)
