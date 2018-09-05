#include <cmath>
#include "sys_potential.h"

double potential(double x){
	return cos(x) + cos(2*x) + cos(3*x);
}

double force(double x){
	return sin(x) + 2*sin(2*x) + 3*sin(3*x);
}

double potential_second_derivative(double x){
	return -cos(x) - 4*cos(2*x) - 9*cos(3*x);
}
