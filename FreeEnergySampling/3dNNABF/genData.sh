#!/bin/bash

#rm *.dat

for i in $(seq 1 1 20)
	do
	order="_noNN_5_gamma0.1_$i"

	./ABF.py wABF$order.dat wABF_Force$order.dat 5 "yes" "no" 0.1 

	# argv[1] = original data, argv[2] = force data, argv[3] = time length, argv[4] = abfflag, argv[5] = nnCheckFlag, argv[6] = frictCoeff

	wait

	./checkBoltzmannDist.py wABF$order.dat 2 wABFHistogram$order.dat

	wait
done

#xmgrace -block  wABF_Force_noNN_5_gamma0.1_1.dat -bxy 1:3 


#xmgrace wABF_Force_yesNN* &
xmgrace wABF_Force_noNN* &

killall -9 genData.sh

