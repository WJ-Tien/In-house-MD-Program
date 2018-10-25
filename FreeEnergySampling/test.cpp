#include <iostream>
#include <vector>

using namespace std;

int main(){
	
	vector<int>test;
	test.push_back(1);
	test.push_back(2);
	cout << test[0] << endl;
	cout << test[1] << endl;
	cout << test.size() << endl;

	return 0;
}	
