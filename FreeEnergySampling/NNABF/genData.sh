#!/bin/bash

for i in $(seq 1 1 1)
do
order="_noNN_50000_gamma1_$i"

./ABF.py wABF$order.dat wABF_Force$order.dat 50000 "no" "no" 1

# argv[1] = original data, argv[2] = force data, argv[3] = time length, argv[4] = abfflag, argv[5] = nnCheckFlag

wait

./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat

wait
done

killall -9 genData.sh

