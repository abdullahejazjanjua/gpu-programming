#include <cstdlib>
#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>
#include <chrono>


#include "matrixAdd.h"

using namespace std;

float* allocateMatrix(int num_rows, int num_cols);
void freeMatrix(float *mat, int num_rows);
float* addMatrix(float *mat1, float *mat2, int num_rows, int num_cols);
void readMatrix(ifstream& f, float *mat, int num_rows, int num_cols);
void printMatrix(float *mat, int num_rows, int num_cols, ostream& out);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "USAGE: ./a.out <input-file>.txt [<output-file>.txt]\num_rows";
        return 1;
    }
    
    ifstream ifile(argv[1]);
    if (!ifile.is_open())
    {
        cerr << "File not open\num_rows";
        return 1;
    }
    string str;
    
    getline(ifile, str);
    int num_rows = stoi(str);
    getline(ifile, str);
    int num_cols = stoi(str);

    float *A = allocateMatrix(num_rows, num_cols);
    float *B = allocateMatrix(num_rows, num_cols);
    
    getline(ifile, str);
    readMatrix(ifile, A, num_rows, num_cols);
    readMatrix(ifile, B, num_rows, num_cols);    
    ifile.close();
    
    float *C = allocateMatrix(num_rows, num_cols);
    auto begin = chrono::high_resolution_clock::now();
    matrixAdd(A, B, C, num_rows, num_cols);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - begin);
    if (argc == 2)
    {
        printMatrix(C, num_rows, num_cols, cout);
        cout << "Time taken: " << duration.count() << endl;
    }
    else
    {
        ofstream ofile(argv[2]);
        
        ofile << num_rows << "\n";
        ofile << num_cols << "\n\n";
        ofile << "Time taken: " << duration.count() << "\n\n";
        printMatrix(C, num_rows, num_cols, ofile);
        ofile.close();
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
}

float* allocateMatrix(int num_rows, int num_cols)
{
    float *mat = new float[num_rows * num_cols]; 
    if (mat == nullptr)
    {
        cerr << "Couldn't allocate memory\n";
        exit(EXIT_FAILURE);
    }
    return mat;
}

void readMatrix(ifstream& f, float *mat, int num_rows, int num_cols)
{
    string str;
    int i = 0;
    while (getline(f, str) && i != num_rows)
    {
        string val;
        int j = 0;
        stringstream ss(str);
        while (getline(ss, val, ' ') && j < num_cols)
        {
            mat[i * num_cols + j] = stof(val);
            j++;
        }
        i++;
    }
}

void printMatrix(float *mat, int num_rows, int num_cols, ostream& out)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
             out << mat[i * num_cols + j] << " ";
        }
        out << endl;
    }
}