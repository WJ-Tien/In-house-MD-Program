CC = g++
CCFLAGS = -std=c++11

run: run.cpp hamiltonian.o  sys_potential.o  verlet_integrator.o langevin_integrator.o monte_carlo.o energy_minimization.o
	$(CC) run.cpp hamiltonian.o sys_potential.o verlet_integrator.o langevin_integrator.o monte_carlo.o energy_minimization.o $(CCFLAGS) -o run

verlet_integrator.o: verlet_integrator.cpp
	$(CC) -c $(CCFLAGS) verlet_integrator.cpp

sys_potential.o: sys_potential.cpp
	$(CC) -c $(CCFLAGS) sys_potential.cpp

hamiltonian.o: hamiltonian.cpp
	$(CC) -c $(CCFLAGS) hamiltonian.cpp

langevin_integrator.o: langevin_integrator.cpp
	$(CC) -c $(CCFLAGS) langevin_integrator.cpp

monte_carlo.o: monte_carlo.cpp
	$(CC) -c $(CCFLAGS) monte_carlo.cpp

energy_minimization.o: energy_minimization.cpp
	$(CC) -c $(CCFLAGS) energy_minimization.cpp

clean:
	rm *.o run 

#-std=c++11 must be applied for specific modules  (random number generator)

