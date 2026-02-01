#include <iostream>
#include <bits/stdc++.h>

#include "vectorAdd.h"

using namespace std;

#define N 100000

int main()
{
    float *A, *B, *C;
    int byteSize = N * sizeof(float);
    
    A = (float *) malloc(byteSize);
    B = (float *) malloc(byteSize);
    C = (float *) malloc(byteSize);
    
    for (int i = 0; i < N; i++)
    {
        float val = (float)(rand()) / (float)(rand());
        A[i] = val;
        B[i] = val;
    }
    
    
    
    vectorAdd(A, B, C, N);
    
    for (int i = 0; i < 10; i++)
    {
        cout << "A[" << i << "] (" << A[i] << ") + " <<  "B[" << i << "] (" << B[i] << ") = " << "C[" << i << "] (" << C[i] << ")" << endl;
    } 
}