#include <cmath>
#include "verlet_integrator.h"

//http://www.eng.buffalo.edu/~kofke/ce530/Lectures/Lecture11/sld041.htm
//http://www.dsf.unica.it/~fiore/vv.pdf
//Velocity Verlet Algorithm

double* verlet_integrator(double current_disp, double current_vel, double current_time, 
                          double tintv, double mass,
                          double (*force)(double), double MD_PARM[], double box_length)
{
	double current_force;
	double next_disp;
	double next_vel;
	double next_time;
	double next_force;

	current_force = (*force)(current_disp);
	next_disp = current_disp + current_vel * tintv + (0.5/mass) * current_force * tintv * tintv;
	next_disp -= ((box_length) * round((next_disp/ box_length))); //PBC
	next_force = (*force)(next_disp);
	next_vel = current_vel + (0.5/mass) * (current_force + next_force) * tintv;
	next_time = current_time + tintv;	

	MD_PARM[0] = next_disp;
	MD_PARM[1] = next_vel;
	MD_PARM[2] = next_time;
	
	return MD_PARM;
	
}
