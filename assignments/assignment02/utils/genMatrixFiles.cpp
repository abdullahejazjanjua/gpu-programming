#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <new>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        cerr << "USAGE: ./a.out <n> <k> <m>\n";
        return 1;
    }
    int n = stoi(argv[1]);
    int k = stoi(argv[2]);
    int m = stoi(argv[3]);
    
    float *mat1 = nullptr, *mat2 = nullptr;
    try {
        mat1 = new float[(size_t) n * k];
    }
    catch(const std::bad_alloc& e) {
        cerr << "Memory Allocation failed\n";
        exit(EXIT_FAILURE);
    }
    
    try {
        mat2 = new float[(size_t) k * m];
    }
    catch(const std::bad_alloc& e) {
        cerr << "Memory Allocation failed\n";
        exit(EXIT_FAILURE);
    }
    
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            mat1[i * k + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            mat2[i * m + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    
    ofstream ofile("../input.txt");
    ofile << n << "\n";
    ofile << k << "\n";
    ofile << m << "\n";
    
    ofile << "\n";
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (j == k - 1)
                ofile << mat1[i * k + j];
            else
                ofile << mat1[i * k + j] << " ";
        }
        ofile << "\n";
    }
    
    ofile << "\n";
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            if (j == m - 1)
                ofile << mat2[i * m + j];
            else
                ofile << mat2[i * m + j] << " ";
        }
        ofile << "\n";
    }
    
    delete [] mat1;
    delete [] mat2;
    
}