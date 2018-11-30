#!/bin/bash

./NN.py
wait

xmgrace -block trainingSet/wABF_Force_test.dat -bxy 1:4 -block analysis/force_result.dat -bxy 1:4 -block early_stage/wABF_Force001.dat -bxy 1:4 &
wait

killall -9 run.sh
