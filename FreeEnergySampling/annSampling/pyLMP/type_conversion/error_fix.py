#!/usr/bin/env python3
from lammps import PyLammps, lammps
import numpy as np
import sys

l = lammps()
lmp = PyLammps(ptr=l)


lmp.units("lj")
lmp.atom_style("atomic")
lmp.lattice("fcc", 0.8442)
lmp.region("box", "block", 0, 4, 0, 4, 0, 4)
lmp.create_box(1, "box")
lmp.create_atoms(1, "box")
lmp.mass(1, 1.0)


lmp.velocity("all", "create", 10, 87287)
lmp.pair_style("lj/cut", 2.5)
lmp.pair_coeff(1, 1, 1.0, 1.0, 2.5)
lmp.neighbor(0.3, "bin")
lmp.neigh_modify("delay", 0, "every", 20, "check no")

lmp.fix("1 all nve")
a = l.extract_fix("2",2,2)
print(a.contents)
#nlocal = l.extract_global("nlocal",0) 
#print(nlocal)
#lmp.fix("2 all addforce 1.0 0.0 0.0")

#f = l.extract_atom("f",3)
#a = np.ctypeslib.as_array(f.contents,shape=(natoms, 3))
#print(a)

lmp.dump("id all atom 50 dumpT10.0.lammpstrj")

lmp.thermo(50)
lmp.run(10000)
