#ifndef _verlet_integrator_H
#define _verlet_integrator_H

double* verlet_integrator(double, double, double,
                          double, double, 
                          double(*)(double), double [], double);

#endif
