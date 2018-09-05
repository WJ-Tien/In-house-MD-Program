#include <iostream>

using namespace std;

double test(double a[]){
	return a[0] + a[1];
}

int main(){
	double b[2] = {1.1,2.2};
	cout << test(b) << endl;

	return 0;
}
