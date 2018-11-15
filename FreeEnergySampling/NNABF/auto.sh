#!/bin/bash

./ABF.py
wait
./checkBoltzmannDist.py long4nn.dat 2 longout.dat
