#!/bin/bash

for i in $(seq 1 1 1)
	do
	order="_m1_T4_yesABF_yesNN_2500_gamma0.1_2_$i"

	./ABF_v2.py wABF$order.dat wABF_Force$order.dat 2500 "yes" "yes" 0.1 

	# argv[1] = original data, argv[2] = force data, argv[3] = time length, argv[4] = abfflag, argv[5] = nnCheckFlag, argv[6] = frictCoeff

	wait

	./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat

	wait
done

./error.py estimate wABF_Force$order.dat wABF_Force_m1_T4_yesABF_noNN_2500_gamma0.1_1.dat wABF_Force_m1_T4_noABF_noNN_2500_gamma0.1_1.dat
wait

killall -9 genData.sh

