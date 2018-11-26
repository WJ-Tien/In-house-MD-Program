#!/bin/bash

order="005"

./ABF.py wABF$order.dat wABF_Force$order.dat 400
# argv[1] = original data, argv[2] = force data, argv[3] = time length
wait
./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat
wait

mv *.dat trainingSet/ 
