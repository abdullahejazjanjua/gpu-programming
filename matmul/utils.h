#ifndef UTILS_H
#define UTILS_H
#endif

typedef struct {
  int rows, cols;
  float *dataptr;
} matrix;

// cuda kernel wrapper
void matmul(float *mat1_h, float *mat2_h, float *mat3_h, int n, int m, int k);

// host code helpers
matrix *generate_matrix(int num_rows, int num_cols = 1);
float* matmul_cpu(float *mat1, float *mat2, float *mat3, int n, int k, int m);
void check_correctness(float *mat1, float *mat2, int n, int k, int m);
void populate_matrix(matrix *mat);
void benchmark(int n, int m, int k);
