#include <iostream>
#include <fstream>

#include "../utils/utils.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "USAGE: ./a.out <input-file>.txt [<output-file>.txt]\n";
        return 1;
    }
    
    ifstream ifile(argv[1]);
    if (!ifile.is_open())
    {
        cerr << "File not open\n";
        return 1;
    }
    string str;
    
    getline(ifile, str);
    int n = stoi(str);
    getline(ifile, str);
    int k = stoi(str);
    getline(ifile, str);
    int m = stoi(str);

    float *A = allocateMatrix(n, k);
    float *B = allocateMatrix(k, m);
    float *C = allocateMatrix(n, m);
    
    getline(ifile, str);
    readMatrix(ifile, A, n, k);
    getline(ifile, str); // consume the empty space
    readMatrix(ifile, B, k, m);    
    ifile.close();

    mulMatrixGPUTiling(A, B, C, n, m, k);
    
    if (argc == 2)
    {
        printMatrix(C, n, m, cout);
    }
    else
    {
        ofstream ofile(argv[2]);
        printMatrix(C, n, m, ofile);
        ofile.close();
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
}