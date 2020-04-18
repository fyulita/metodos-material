#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <math.h>

using namespace std;

// http://www.cplusplus.com/reference/climits/

int main(){
	
	float a = INT_MAX; // 2147483647
	float b = LONG_MAX; // 9223372036854775807
	
	float f = a + b;

	if (f == LONG_MAX){
		cout << "Siempre ganan los grandes!" << endl;
	}else{
		cout << "A veces tambien gana San Lorenzo" << endl;
	}

	return 0;
}
