#ifndef _langevin_integrator_H
#define _langevin_integrator_H

double* langevin_integrator(double , double, double, 
                            double , double, 
                            double (*)(double), double[], double,
                            double, double);

#endif


