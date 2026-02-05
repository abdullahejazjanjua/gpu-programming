#include <stdio.h>

__global__ void matmulKernel(float *mat1, float *mat2, float *mat3, int n, int m, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    
    if (row < n && col < m) 
    {
        int idx = row * m + col; 
        float val = 0;
        
        for (int i = 0; i < k; i++) 
            val += (mat1[row * k  + i] * mat2[i * m + col]); 
        mat3[idx] = val;
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

void matmul(float *mat1_h, float *mat2_h, float *mat3_h, int n, int m, int k)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *mat1_d = matrixAllocate(n, k);
    float *mat2_d = matrixAllocate(k, m);
    float *mat3_d = matrixAllocate(n, m);
    
    matrixCopy(mat1_d, mat1_h, n, k, cudaMemcpyHostToDevice);
    matrixCopy(mat2_d, mat2_h, k, m, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(m/32.0), ceil(n/32.0), 1);
    cudaEventRecord(start);
    matmulKernel<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, n, m, k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    matrixCopy(mat3_h, mat3_d, n, m, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("GPU TIME: %f microsecs\n", milliseconds * 1000);
    
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(mat3_d);
}