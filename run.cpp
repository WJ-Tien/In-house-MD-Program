#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "energy_minimization.h"
#include "verlet_integrator.h"
#include "langevin_integrator.h"
#include "monte_carlo.h"
#include "sys_potential.h"
#include "hamiltonian.h"

using namespace std;

int main(int argc, char *argv[]){

	ifstream file_PARM;
	ofstream file_OUT;
	file_PARM.open(argv[1]);
	file_OUT.open(argv[2]);

	int	first_frame = 0;
	int count_lines = 0;
	double *s;

	string read_parm;
	string parm_name;
	string equal_sign;
	string mode;

	while(file_PARM >> parm_name >> equal_sign >> read_parm){ //count_line and read mode
		if (parm_name[0] != ';') 
			count_lines++;
		if ((parm_name[0] != ';') && (parm_name.compare("mode") == 0))
			mode = read_parm;	
	}	

	file_PARM.clear();
	file_PARM.seekg(0);

	if (mode.compare("em_gradient_descent") == 0){
		count_lines += 2;
	} else {;}

	double MD_PARM[count_lines - 1] = {0.};

	if (mode.compare("em_gradient_descent") == 0){

		while (file_PARM >> parm_name >> equal_sign >> read_parm){ //read PARM file

			if ((parm_name[0] != ';') && (parm_name.compare("initial_displacement") == 0)){
				MD_PARM[0] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("initial_step") == 0)){
				MD_PARM[1] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("total_steps") == 0)){
				MD_PARM[2] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("box_length") == 0)){
				MD_PARM[3] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("optimization_scalar") == 0)){
				MD_PARM[4] = atof(read_parm.c_str());
			}
		}
		
		do {

			cout << "Frame" << " " << first_frame <<endl;
			cout << "Steps" << " " << MD_PARM[1] <<endl;
			cout << "Displacement" << " " << MD_PARM[0] <<endl;
			cout << "Energy_Convergence" << " " << MD_PARM[5] << endl;
			cout << endl;

			file_OUT << first_frame << " " << MD_PARM[1] << " " << MD_PARM[0] << " " <<  MD_PARM[5] << " " << endl;

			if (MD_PARM[6] == 1.0) {break;}

			s = gradient_descent(MD_PARM[0], MD_PARM[1], MD_PARM[2], potential, force, 
                           MD_PARM[4], MD_PARM, MD_PARM[3], potential_second_derivative);
													
			MD_PARM[0] = s[0];
			MD_PARM[1] = s[1];
			MD_PARM[5] = s[5];
			MD_PARM[6] = s[6];

			first_frame++ ;

		} while (MD_PARM[1] <= MD_PARM[2]);
	}

	if (mode.compare("newton") == 0){

		while (file_PARM >> parm_name >> equal_sign >> read_parm){ //read PARM file

			if ((parm_name[0] != ';') && (parm_name.compare("initial_displacement") == 0)){
				MD_PARM[0] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("initial_velocity") == 0)){
				MD_PARM[1] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("initial_time") == 0)){
				MD_PARM[2] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("time_interval") == 0)){
				MD_PARM[3] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("time_length") == 0)){
				MD_PARM[4] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("mass") == 0)){
				MD_PARM[5] = atof(read_parm.c_str());
			}
			if ((parm_name[0] != ';') && (parm_name.compare("box_length") == 0)){
				MD_PARM[6] = atof(read_parm.c_str());
			}
		}
		
		do {

			cout << "Frame" << " " << first_frame <<endl;
			cout << "Time" << " " << MD_PARM[2] <<endl;
			cout << "Displacement" << " " << MD_PARM[0] <<endl;
			cout << "Velocity" << " " << MD_PARM[1]  <<endl;
			cout << "Hamiltonian" << " " << hamiltonian(MD_PARM[0], MD_PARM[1], MD_PARM[5], potential) <<endl;
			cout << endl;

			file_OUT << first_frame << " " << MD_PARM[2] << " " << MD_PARM[0] << " " <<  MD_PARM[1] << " "
							 << hamiltonian(MD_PARM[0], MD_PARM[1], MD_PARM[5], potential) << endl;

			s = verlet_integrator(MD_PARM[0],MD_PARM[1],MD_PARM[2],MD_PARM[3], MD_PARM[5],force, 
                            MD_PARM, MD_PARM[6]);
													
			MD_PARM[0] = s[0];
			MD_PARM[1] = s[1];
			MD_PARM[2] = s[2];

			first_frame++ ;

		} while (MD_PARM[2] <= MD_PARM[4] + MD_PARM[3]/2);
	}

	if (mode.compare("langevin") == 0){

		while(file_PARM >> parm_name >> equal_sign >> read_parm){ //read PARM file

			if((parm_name[0] != ';') && (parm_name.compare("initial_displacement") == 0)){
				MD_PARM[0] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("initial_velocity") == 0)){
				MD_PARM[1] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("initial_time") == 0)){
				MD_PARM[2] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("time_interval") == 0)){
				MD_PARM[3] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("time_length") == 0)){
				MD_PARM[4] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("mass") == 0)){
				MD_PARM[5] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("box_length") == 0)){
				MD_PARM[6] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("temperature") == 0)){
				MD_PARM[7] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("friction_coeff") == 0)){
				MD_PARM[8] = atof(read_parm.c_str());
			}
		}

		do {

			cout << "Frame" << " " << first_frame <<endl;
			cout << "Time" << " " << MD_PARM[2] <<endl;
			cout << "Displacement" << " " << MD_PARM[0] <<endl;
			cout << "Velocity" << " " << MD_PARM[1]  <<endl;
			cout << "Hamiltonian" << " " << hamiltonian(MD_PARM[0], MD_PARM[1], MD_PARM[5], potential) <<endl;
			cout << endl;

			file_OUT << first_frame << " " << MD_PARM[2] << " " << MD_PARM[0] << " " <<  MD_PARM[1] << " "
							 << hamiltonian(MD_PARM[0], MD_PARM[1], MD_PARM[5], potential) << endl;

			s = langevin_integrator(MD_PARM[0],MD_PARM[1],MD_PARM[2],MD_PARM[3], MD_PARM[5],force, 
                            MD_PARM, MD_PARM[6], MD_PARM[7], MD_PARM[8]);
													
			MD_PARM[0] = s[0];
			MD_PARM[1] = s[1];
			MD_PARM[2] = s[2];

			first_frame++ ;

		} while (MD_PARM[2] <= MD_PARM[4] + MD_PARM[3]/2);
	}

	if (mode.compare("monte_carlo") == 0){

		while(file_PARM >> parm_name >> equal_sign >> read_parm){ //read PARM file

			if((parm_name[0] != ';') && (parm_name.compare("initial_displacement") == 0)){
				MD_PARM[0] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("initial_time") == 0)){
				MD_PARM[1] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("time_interval") == 0)){
				MD_PARM[2] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("time_length") == 0)){
				MD_PARM[3] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("box_length") == 0)){
				MD_PARM[4] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("temperature") == 0)){
				MD_PARM[5] = atof(read_parm.c_str());
			}
			if((parm_name[0] != ';') && (parm_name.compare("artificial_displacement") == 0)){
				MD_PARM[6] = atof(read_parm.c_str());
			}
		}

		do {
			cout << "Frame" << " " << first_frame <<endl;
			cout << "Time" << " " << MD_PARM[1] <<endl;
			cout << "Displacement" << " " << MD_PARM[0] <<endl;
			cout << "Potential Energy" << " " << potential(MD_PARM[0]) <<endl;
			cout << endl;

			file_OUT << first_frame << " " << MD_PARM[1] << " " << MD_PARM[0] << " " << potential(MD_PARM[0]) 
               << endl;

			s = monte_carlo(MD_PARM[0],MD_PARM[1], potential, MD_PARM[2], MD_PARM[6],
                      MD_PARM, MD_PARM[4], MD_PARM[5]);
													
			MD_PARM[0] = s[0];
			MD_PARM[1] = s[1];

			first_frame++ ;

		} while (MD_PARM[1] <= MD_PARM[3] + MD_PARM[2]/2);
	}

  // new implementation here

	file_PARM.close();
	file_OUT.close();

	return 0;
}
