#include <iostream>
#include "utils/utils.h"

using namespace std;



int main(int argc, char *argv[]) {
    cout << "CPU vs GPU-naive vs GPU-tiled" << endl << endl;

    int sizes[][3] = {
        {128, 128, 128}, 
        {256, 256, 256}, 
        {512, 512, 512}, 
        {1024, 1024, 1024},
        {128, 256, 512}, 
        {512, 128, 256}, 
        {256, 512, 1024},
        {1024, 512, 256},
        {100, 500, 200}, 
        {500, 200, 1000}, 
        {200, 1000, 500},
        {2048, 1024, 512}, 
        {4096, 256, 1024},
        {300, 600, 300}, 
        {600, 300, 900}
    };
    int size = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < size; i++) {
        int n = sizes[i][0];
        int m = sizes[i][1];
        int k = sizes[i][2];
        
        cout << "mat1: " << n << " x " << k << " mat2: " << k << " x " << m << endl;
        benchmark(n, m, k);
        cout << endl;
    }   
}

