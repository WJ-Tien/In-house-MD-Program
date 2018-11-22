#!/bin/bash

./ABF.py
wait
./checkBoltzmannDist.py wABF000.dat 2 wABFHistogram000.dat
