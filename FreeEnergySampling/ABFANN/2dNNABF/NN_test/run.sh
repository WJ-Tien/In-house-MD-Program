#!/bin/bash
for i in $(seq 1 1 30)
do
	./NN.py
	wait
done
		
