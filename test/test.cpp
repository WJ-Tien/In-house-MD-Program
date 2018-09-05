#include <iostream>
using namespace std;

double test2(double x){
	return x+1;
}

double test(double x){
	return x*20;
}

double force(double (*x) (double)){
	//double (*i) (double);
	//i = x;
	return (*x)(10);
}

int main(){

	//double (*f)(int);
	//double (*j) (double);	
	cout<< force(test) <<endl;
	cout<< force(test) <<endl;
	cout<< force(test) <<endl;

	return 0;
}

