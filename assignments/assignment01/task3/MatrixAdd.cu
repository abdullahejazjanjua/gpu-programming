#include <stdio.h>
__global__ void matrixAddKernel(float *mat1, float *mat2, float *mat3, int num_rows, int num_cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    
    if (i < num_rows && j < num_cols)
    {
        int global_idx = i * num_cols + j;
        mat3[global_idx] = mat1[global_idx] + mat2[global_idx];
    }
}


void matrixCopy(float *matdest, float *matsrc, int num_rows, int num_cols, cudaMemcpyKind direction)
{
    cudaError_t err = cudaMemcpy(matdest, matsrc, (num_rows * num_cols * sizeof(float)), direction);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to copy matdest from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

float* matrixAllocate(int num_rows, int num_cols)
{
    float *mat;
    cudaError_t err = cudaMalloc((void **)&mat, (num_rows * num_cols * sizeof(float)));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to allocate memory in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
     }
    return mat;
}

void matrixAdd(float *mat1_h, float *mat2_h, float *mat3_h, int num_rows, int num_cols)
{
    float *mat1_d = matrixAllocate(num_rows, num_cols);
    float *mat2_d = matrixAllocate(num_rows, num_cols);
    float *mat3_d = matrixAllocate(num_rows, num_cols);
    
    matrixCopy(mat1_d, mat1_h, num_rows, num_cols, cudaMemcpyHostToDevice);
    matrixCopy(mat2_d, mat2_h, num_rows, num_cols, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16, 1); // x (columns), y (rows), z (depth)
    dim3 gridDim(ceil(num_cols/16), ceil(num_rows/16), 1); // x (columns), y (rows), z (depth)
    matrixAddKernel<<<gridDim, blockDim>>>(mat1_d, mat2_d, mat3_d, num_rows, num_cols);
    
    matrixCopy(mat3_h, mat3_d, num_rows, num_cols, cudaMemcpyDeviceToHost);
}