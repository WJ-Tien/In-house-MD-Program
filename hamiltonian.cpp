#include "hamiltonian.h"

double hamiltonian(double current_disp, double current_vel, double mass, double (*pot)(double)){
	return 0.5*mass*current_vel*current_vel + (*pot)(current_disp);
}

