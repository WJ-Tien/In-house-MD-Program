#include <cmath>
#include "energy_minimization.h"
#include <random>

double* gradient_descent(double current_disp, int current_step, int total_steps,  double (*pot)(double), double (*force)(double), double opt_scalar, double MD_PARM[], double box_length,
                         double (*potential_second_derivative)(double)){

	double next_disp;	
	double randnum;
	double convexityCheck_First_Order;
	double convexityCheck_Second_Order;

	next_disp = current_disp + opt_scalar * (*force)(current_disp); 
	next_disp -= (round(next_disp / box_length) * box_length);
	current_step ++;
	convexityCheck_First_Order = -(*force)(next_disp);
	convexityCheck_Second_Order = (*potential_second_derivative)(next_disp);

	if (convexityCheck_Second_Order < 0.0 && convexityCheck_First_Order == 0.0){ //To escape from the "fake" local minimum

		std::random_device rd;
		randnum = (double) rd() / rd.max();

		if (randnum < 0.5)
			next_disp -= 0.1;
		else 
			next_disp += 0.1;
	}
	
	if ((next_disp - current_disp < 0.01) && ((*pot)(next_disp) - (*pot)(current_disp) < 0.01) && (std::abs((*force)(next_disp)) < 0.01)){
		MD_PARM[0] = next_disp;
		MD_PARM[1] = current_step; 
		MD_PARM[5] = (*pot)(next_disp) - (*pot)(current_disp);
		MD_PARM[6] = 1.0; 
		return MD_PARM; 
	}

	else { 
		MD_PARM[0] = next_disp;
		MD_PARM[1] = current_step; 
		MD_PARM[5] = (*pot)(next_disp) - (*pot)(current_disp);
		MD_PARM[6] = 0.0;
		return MD_PARM;
	}


}
