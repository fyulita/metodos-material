#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>

using namespace std;

int main(){
    int cantidad;
    cin >> cantidad;
    vector<float> numeros;
    for(int i=0;i<cantidad;i++){
        numeros.push_back(pow(2,-20));
    }

    auto start = chrono::steady_clock::now();
    float suma = 0.0;
    for(int i=0;i<cantidad;i++){
        suma += numeros[i];
    }
    auto end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << endl;
    
    return 0;
}


	
