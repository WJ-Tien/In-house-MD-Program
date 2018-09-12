#include <cmath>
#include <random>
#include <chrono>
#include "langevin_integrator.h" 

//http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf   (P21, Eq98)
//https://en.wikipedia.org/wiki/Langevin_dynamics
//http://www.cplusplus.com/reference/random/normal_distribution/normal_distirbution/
//use normal distribution generator to generate random_theta and random_xi

double* langevin_integrator(double current_disp, double current_vel, double current_time, 
                            double tintv, double mass,
                            double (*force)(double), double MD_PARM[], double box_length,
                            double temperature, double frictCoeff)
{
	double next_disp;
	double next_vel;
	double next_time;
	double current_force;
	double next_force;
	double Ct;
	double kb = 1;
	//double kb = 1.38064852e-23; // J/K
	double sigma;
	double random_theta, random_xi;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::default_random_engine generator (seed);
	std::normal_distribution <double> distribution (0.0,1.0);
	random_theta = distribution(generator);
	random_xi = distribution(generator);
	
	sigma = sqrt(2.0 * kb * temperature * frictCoeff / mass);

	current_force = (*force)(current_disp);	

	Ct = (0.5 * tintv * tintv * (current_force - frictCoeff * current_vel) / mass) + 
       sigma * pow(tintv, 1.5) * (0.5 * random_xi + 0.288675 * random_theta);
	
	next_disp = current_disp + tintv * current_vel + Ct;
	next_disp -= (round(next_disp / box_length) * box_length); //PBC
	next_force = (*force)(next_disp);
	next_vel = current_vel + (0.5 * tintv * (next_force + current_force) / mass) -
             tintv * frictCoeff * current_vel + sigma * sqrt(tintv) * random_xi - 
             frictCoeff * Ct; 
		
	next_time = current_time + tintv;

	MD_PARM[0] = next_disp;
	MD_PARM[1] = next_vel;
	MD_PARM[2] = next_time;
	
	return MD_PARM;
}
