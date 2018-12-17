#!/bin/bash

for i in $(seq 1 1 1)
	do
	order="_m2_T6_yesABF_noNN_5000_gamma0.1_$i"

	./ABF_v2.py wABF$order.dat wABF_Force$order.dat 5000 "yes" "no" 0.1 

	# argv[1] = original data, argv[2] = force data, argv[3] = time length, argv[4] = abfflag, argv[5] = nnCheckFlag, argv[6] = frictCoeff

	wait

	./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat

	wait
done

#./error.py estimate wABF_Force_yesABF_yesNN_2500_gamma0.1_1.dat wABF_Force_m2_T6_yesABF_noNN_2500_gamma0.1_1.dat wABF_Force_m2_T6_noABF_noNN_2500_gamma0.1_1.dat
wait

killall -9 genData.sh

