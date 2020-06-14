#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;

int main(){
	
	float div = .26f;
	
	printf("%d\n",(int)(3250*div));
	printf("%d\n",(int)(3250*0.26));
	printf("%d\n",(int)(3250*26)/100);
	printf("%d\n",(int)(3250.0*0.26f));
	return 0;
}
