#!/bin/bash

for i in $(seq 1 1 30)
do
	./NN_1D.py 0.075
	wait
	mv out out$i
done

#xmgrace estimate out* &
xmgrace Force_m100_T0.001_noABF_noNN_TL_50000.dat out* &

		
