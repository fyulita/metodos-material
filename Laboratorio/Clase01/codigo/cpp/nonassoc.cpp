#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>

using namespace std;

int main(){

	float a = 1.0e23;
	float b = -1.0e23;
	float c = 1.0;

	printf("%f\n", (a+b)+c);
	printf("%f\n", a+(b+c));

	return 0;
}
