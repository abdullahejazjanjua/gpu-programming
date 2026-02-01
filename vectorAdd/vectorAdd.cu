#include <stdio.h>

__global__ void vectorAddKernel(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(float *A_h, float *B_h, float *C_h, int N)
{
    float *A_d, *B_d, *C_d;
    int byteSize = N * sizeof(float);
    
    cudaMalloc((void **)&A_d, byteSize);
    cudaMalloc((void **)&B_d, byteSize);
    cudaMalloc((void **)&C_d, byteSize);
    
    cudaMemcpy(A_d, A_h, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, byteSize, cudaMemcpyHostToDevice);
    
    dim3 dimGrid(ceil(N/256), 1, 1) // x, y, z
    dim3 dimBlock(ceil(256, 1, 1)) // x, y, z
    vectorAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);
    
    cudaMemcpy(C_h, C_d, byteSize, cudaMemcpyDeviceToHost);
    
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}