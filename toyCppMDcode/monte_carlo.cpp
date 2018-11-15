#include <cmath>
#include <random>
#include <algorithm>
#include "monte_carlo.h"

//https://web.northeastern.edu/afeiguin/phys5870/phys5870/node80.html
//https://web.stanford.edu/class/cs279/lectures/lecture4.pdf (P35)
//https://blog.gtwang.org/programming/cpp-random-number-generator-and-probability-distribution-tutorial/
//https://zh.wikipedia.org/wiki/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95

double* monte_carlo(double current_disp, double current_time, 
										double (*pot)(double), double tintv, double artificial_disp,
										double MD_PARM[], double box_length, 
										double temperature)
{
	double randnum;
	double current_pot;
	double next_disp;
	double next_time;
	double next_pot;
	double s;
	double kb = 1;
	double boltzmann_factor;
	double acceptance_ratio;
	
	std::random_device rd;
	randnum = (double) rd() / rd.max();

	current_pot = (*pot)(current_disp);
	next_disp = current_disp + artificial_disp * (randnum - 0.5);
	next_disp -= (round(next_disp/box_length) * box_length);
	next_pot = (*pot)(next_disp);
	next_time = current_time + tintv;

	boltzmann_factor = exp(-(next_pot - current_pot)/(kb*temperature));
	acceptance_ratio = std::min(1.0, boltzmann_factor);

	if (acceptance_ratio == 1.0){
		MD_PARM[0] = next_disp;
		MD_PARM[1] = next_time;
		return MD_PARM;	
	}
	
	else {
		if (randnum < boltzmann_factor){
			MD_PARM[0] = next_disp;
			MD_PARM[1] = next_time;
			return MD_PARM;	
		}
		
		else {
			MD_PARM[0] = current_disp;
			MD_PARM[1] = next_time;
			return MD_PARM;
		}
	}

}
