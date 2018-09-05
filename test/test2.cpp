#include <iostream>
using namespace std;

double* test(double x){
	//double y[2] = {0.0,0.0};
	double *y = new double(2*sizeof(double));
	y[0] = x+3;
	y[1] = x+2;
	
	return y;
}

int main(){
	cout << *test(0) << endl;
	cout << *(test(0) + 1)  << endl;
	return 0;
}

