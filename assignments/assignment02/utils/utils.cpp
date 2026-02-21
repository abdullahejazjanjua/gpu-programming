#include <cstdlib>
#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "utils.h"

using namespace std;

float* allocateMatrix(int num_rows, int num_cols) {
    float *mat = new float[num_rows * num_cols]; 
    if (mat == nullptr) {
        cerr << "Couldn't allocate memory\n";
        exit(EXIT_FAILURE);
    }
    return mat;
}

void readMatrix(ifstream& f, float *mat, int num_rows, int num_cols) {
    string str;
    int i = 0;
    while (getline(f, str) && i != num_rows) {
        string val;
        int j = 0;
        stringstream ss(str);
        while (getline(ss, val, ' ') && j < num_cols) {
            mat[i * num_cols + j] = stof(val);
            j++;
        }
        i++;
    }
}

void printMatrix(float *mat, int num_rows, int num_cols, ostream& out) {
    out << "size: " << num_rows << " x " << num_cols << endl;
    out << std::fixed << std::setprecision(2);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
             out << mat[i * num_cols + j] << " ";
        }
        out << endl;
    }
}

float* mulMatrixCPU(float *mat1, float *mat2, int N, int M, int K) {
    // mat1: N, K   mat2: K, M
    float *mat3 = allocateMatrix(N, M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float val = 0.0;
            for (int k = 0; k < K; k++) {
                val += (mat1[i * K + k] * mat2[k * M + j]);
             }
            mat3[i * M + j] = val;
        }
    }
    
    return mat3;
}