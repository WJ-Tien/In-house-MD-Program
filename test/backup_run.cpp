#include <iostream>
#include <fstream>
#include "verlet_integrator.h"
#include "sys_potential.h"
#include "hamiltonian.h"

#define MY_PI 3.1415926

using namespace std;

int main(int argc, char *argv[]){

	fstream file;
	file.open("Simulation2.dat", ios::out);

	//Intialize         disp, vel, time, force, tintv, total_time, mass, box_length
	double MD_DATA[8] = {0.0, 10.0, 0.0 , 0.0,   0.2,    10.0,     1.0,   2*MY_PI};
	double t = 0.0;
	int frame = 0;
	double *s;

	//Perform Verlet Algorithm
	cout << "Initial Frame " << 0 << endl;
	cout << "Time " << MD_DATA[2] << endl;
	cout << "Position " << MD_DATA[0] <<endl;
	cout << "Velocity " << MD_DATA[1] << endl;
	cout << "Mass " << MD_DATA[6] << endl;
	cout << "Hamiltonian " << hamiltonian(MD_DATA[0], MD_DATA[1], MD_DATA[6], potential) << endl;
	cout << endl;
  file << MD_DATA[2] << " " << MD_DATA[0] << " " <<  MD_DATA[1] << " " << hamiltonian(MD_DATA[0], MD_DATA[1], MD_DATA[6], potential) << endl;

	frame ++ ;
	
	while (t < MD_DATA[5]-0.02){
		s = verlet_integrator(MD_DATA[0], MD_DATA[1], MD_DATA[2], MD_DATA[3], MD_DATA[4], MD_DATA[6], force, MD_DATA, 2*MY_PI);

		MD_DATA[0] = s[0];
		MD_DATA[1] = s[1];
		MD_DATA[2] = s[2];
		MD_DATA[3] = s[3];

		cout << "Current Frame " << frame << endl;
		cout << "Time " << MD_DATA[2] << endl;
		cout << "Position " << MD_DATA[0] <<endl;
		cout << "Velocity " << MD_DATA[1] << endl;
		cout << "Mass " << MD_DATA[6] << endl;
		cout << "Hamiltonian " << hamiltonian(MD_DATA[0], MD_DATA[1], MD_DATA[6], potential) << endl;
		cout << endl;

		file << MD_DATA[2] << " " << MD_DATA[0] << " " <<  MD_DATA[1] << " " << hamiltonian(MD_DATA[0], MD_DATA[1], MD_DATA[6], potential) << endl;

		frame++ ;
		t = MD_DATA[2];

	}
	//The end of Verlet Algorithm

	return 0;
}
