#!/bin/bash

order="001"

./ABF.py wABF$order.dat wABF_Force$order.dat 10 

# argv[1] = original data, argv[2] = force data, argv[3] = time length

wait

./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat

wait

mv *.dat early_stage/ 
