#include <iostream>
using namespace std;

int stuff() {
    int D = 20;
    int s = D % 12;
    std::cout << s;
    return 1;
}

int main() {
    int x = 5;
    int assault = 10;
    if(x == 10) { // This one here is first
        x = 6;
    } else if(x == 5) { // This one is burried beneath the other if statement
        x = 7;
    } else {
        x = 0;
        assault = 1;
    }
    x+=1;
    // x+=1;
    // x--;
    // x+=2;
    // x=x+2;
    stuff(); stuff();
    return 0;
}
