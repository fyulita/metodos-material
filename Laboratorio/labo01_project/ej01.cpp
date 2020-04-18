# include <cstdio>

using namespace std;

int main(){
    float div = .26f;

    printf("%d\n",(int)(3250 * div));
    printf("%d\n",(int)(3250 * 0.26));
    printf("%d\n",(int)((3250 * 26) / 100));
    printf("%d\n",(int)(3250.0 * 0.26f));
    return 0;
}

// La cuenta posta da 845 pero como convertimos a float y a int el error se propaga. C++ por default castea a double.
