# include <iostream>

using namespace std;

int main(){
    float a = 3.0 / 7.0;
    float eps = 0.000000001;

    cout << abs(a - 3.0 / 7.0) << endl;
    if (abs(a - 3.0 / 7.0) < eps){
        cout << "Seguro entra por aca" << endl;
    } else{
        cout << "Es que es un 7 magico" << endl;
    }
    return 0;
}

