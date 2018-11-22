#!/bin/bash

./ABF.py
wait
./checkBoltzmannDist.py wABF_test.dat 2 wABFHistogram_test.dat
