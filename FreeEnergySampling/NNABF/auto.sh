#!/bin/bash

./ABF.py
wait
./checkBoltzmannDist.py wABF_test2.dat 2 wABFHistogram_test2.dat
