#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace std;

int main(){
	ifstream fin;
	fin.open("test.txt");
	string first;
	string equal;
	string value;
	double MD[1] = {0.};
	double check;
	
	while(fin >> first >> equal >> value){
    if ((first[0] != ';') && (first.compare("bcc") == 0)){
   // if (first[0] != ';') {
			cout << first << endl;
			check = atof(value.c_str()) + 1;
			cout << check<< endl;
			MD[0] = check;
		}
	}
	cout << MD[0] << endl;
	return 0;
}
