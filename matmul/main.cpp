#include <iostream>
#include "utils.h"

using namespace std;

int main() 
{
    int sizes[10][3] = {
        {128, 256, 512},
        {512, 128, 256},
        {256, 512, 1024},
        {1024, 512, 256},
        {100, 500, 200},
        {500, 200, 1000},
        {2048, 1024, 512},
        {4096, 256, 1024},
        {300, 600, 300},
        {600, 300, 900}
    };
    
    for (int i = 0; i < 10; i++)
    {
        int n = sizes[i][0];
        int m = sizes[i][1];
        int k = sizes[i][2];
        
        cout << "mat1: " << n << " x " << k << " mat2: " << k << " x " << m << endl;
        benchmark(n, m, k);
        cout << endl;
    }
    
}