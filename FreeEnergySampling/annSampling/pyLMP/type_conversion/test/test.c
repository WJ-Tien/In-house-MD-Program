#include <stdio.h>

struct Ball {
    char color[10];
    double radius;
		double *j;
};

int main() {

	struct Ball ball = {"red", 4.0, NULL};
	struct Ball *ptr;
	ptr = &ball;
	double a = 1000;
	ptr->j = &a;

	printf("%f\n", ptr->radius);
	printf("%s\n", ptr->color);
	printf("%f\n", *ptr->j);


	return 0;
}

