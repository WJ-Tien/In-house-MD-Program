#include <iostream>
#include <random>
#include <chrono>

int main(){
	int i;
	double acc1 = 0;
	double acc2 = 0;
	double acc3 = 0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::normal_distribution <double> distribution (0.0,1.0);
	for(i=0;i<10000;i++){
		double c = distribution(generator);
		double d = distribution(generator);
		acc1 += c*c;	
		acc2 += d*d;
		acc3 += c*d;
	}
	std::cout << acc3/10000  <<" " << acc1/10000 << " " << acc2/10000 << std::endl;
	return 0;
}
