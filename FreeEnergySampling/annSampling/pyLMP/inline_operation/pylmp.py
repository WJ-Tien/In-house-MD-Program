#!/usr/bin/env python3
from lammps import lammps

def post_force_callback(lmp, v):
	L = lammps(ptr=lmp)
	t = L.extract_global("ntimestep", 0)
	print("### POST_FORCE ###", t)

