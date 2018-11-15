## Class importanceSampling (python3 version= testing code) 
1. Method "withFiction_sysPotential" is on the basis of "sysPotential"
2. Method "ABF" are on the basis of "sysForce", "Jacobian", "inverseGradient"

## Nomenclature
1. LDABF = Langevin Dynamics Adaptive Bias Force (Method)
2. bzdistCheck = boltzmann distribution check
3. gamma = friction coefficient
4. TL = time length (simulation time length)
5. temp = temperature
6. w = with; wo = without
7. k = spring Constant

## Note 
1. Dimensionless kb = 1 
2. Higher gamma, better sampling
3. The point on pi must be discarded due to PBC condition (Boltzmann Distribution)
4. For eABF, higher k(springConst), better coupling between Cartcoord and ficcoord 

## Equipartition theorem
1. For three dimensions molecular dynamics, the kinetic energy equals to 3/2 * kb * T
2. For one dimnesion molecular dynamics, the kinetic energy equals to 1/2 * kb * T


## oldest paper for neural network on PES or free energy 
1. https://aip.scitation.org/doi/pdf/10.1063/1.469597?class=pdf (1995)
2. http://www.eoht.info/page/Energy+landscape (difference PES FES)
3. https://pubs.acs.org/doi/pdf/10.1021/jp0680544 (difference PES FES)
4. https://www.researchgate.net/post/What_is_the_difference_between_potential_energy_surface_and_free_energy_surface

## Compile Colvars
1. make -f Makefile-g++
   make yes-USER-COLVARS
   make serial (re-compile)
2. ~/src/LAMMPS/lammps-16Mar18/lib (how to install and compile)

3. ~/src/LAMMPS/lammps-16Mar18/src/lmp_serial -in in.run

## Compile lammps and gromacs
1. For lammps ---> g++ && openmpi/mpich (one of them) && fftw are needed
              ---> src/MAKE/MINE/makefile_mod
   The makefile is copied from OPTIONS and modified by users 

2. make makefile_mod

3. Add lmp_executable to ~/bin or ~/.bashrc
