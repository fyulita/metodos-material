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
    float kahan = 0.0;
    float c = 0.0;
    for(int i=0;i<cantidad;i++){
        float y = numeros[i]-c;
        float t = kahan + y;
        c = (t - kahan) - y;
        kahan = t;
    }
    auto end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << endl;
    
    return 0;
}
