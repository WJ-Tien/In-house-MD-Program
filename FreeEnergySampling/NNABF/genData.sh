#!/bin/bash

order="noNN_long_gamma0.01_3"

./ABF.py wABF$order.dat wABF_Force$order.dat 15000 "yes" "no"

# argv[1] = original data, argv[2] = force data, argv[3] = time length, argv[4] = abfflag, argv[5] = nnCheckFlag

wait

./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat

wait

killall -9 genData.sh
#xmgrace -block analysis/force_result.dat -bxy 1:4 &
#xmgrace -block wABF_Force$order.dat -bxy 1:4 &
#xmgrace wABFHistogramnoNN.dat wABFHistogram$order.dat  &
#codeTests/wABFHistogram_test.dat & 

