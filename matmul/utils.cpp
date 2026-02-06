#include <iostream>
#include <chrono>

using namespace std;

#include "utils.h"

matrix *generate_matrix(int num_rows, int num_cols) 
{
  matrix *mat = new matrix;

  mat->dataptr = new float[num_rows * num_cols];
  if (mat->dataptr == nullptr)
  {
      cerr << "Failed to allocated matrix of size " << num_rows << " x "  << num_cols << "\n";
      delete mat;
      exit(EXIT_FAILURE);
  }
  mat->rows = num_rows;
  mat->cols = num_cols;

  return mat;
}

float* matmul_cpu(float *mat1, float *mat2, float *mat3, int n, int k, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float val = 0.0;
            for (int c = 0; c < k; c++)
            {
                val += (mat1[i * k + c] * mat2[c * m + j]);
            }
            mat3[i * m + j] = val;
        }
    }
    return mat3;
}

void check_correctness(float *mat1, float *mat2, int n, int k, int m)
{
    float err = 0;
    float eps = 1e-8;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float val1 = mat1[i * m + j];
            float val2 = mat2[i * m + j];
            err += abs(val1 - val2) / (max(max(val1, val2), eps));
        }
    }
    cout << "Difference: " << err / (n * m) << endl;
}

void populate_matrix(matrix *mat)
{
    for (int i = 0; i < (mat->rows * mat->cols); i++)
        mat->dataptr[i] = (float)(rand()) / (float)(RAND_MAX);
}


void benchmark(int n, int m, int k)
{
    matrix *mat1 = generate_matrix(n, k); 
    matrix *mat2 = generate_matrix(k, m);
    matrix *mat3_d = generate_matrix(n, m);
    matrix *mat3_h = generate_matrix(n, m);
    
    populate_matrix(mat1);
    populate_matrix(mat2);
    
    matmul(mat1->dataptr, mat2->dataptr, mat3_d->dataptr , mat3_d->rows,  mat3_d->cols, mat1->cols);
    
    auto begin = chrono::high_resolution_clock::now();
    matmul_cpu(mat1->dataptr, mat2->dataptr, mat3_h->dataptr, n, k ,m);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - begin);
    
    cout << "CPU TIME: " << duration.count() << " microsecs" << endl;
    
    check_correctness(mat3_d->dataptr, mat3_h->dataptr, n, k, m);
}