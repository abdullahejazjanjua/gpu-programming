#include "../utils/cuda_error.h"
#include <device_launch_parameters.h>
#include <driver_types.h>

__global__ void matmulKernel(float *mat1, float *mat2, float *mat3,  int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        float val = 0.0;
        for (int k = 0; k < K; k++) {
            val += (mat1[row * K + k] * mat2[k * M + col]);
        }
        mat3[row * M + col] = val;
    }
}

void mulMatrixGPU(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K) {
    float *mat1_d, *mat2_d, *mat3_d;
    gpuErrchk( cudaMalloc((void**)&mat1_d, (N * K * sizeof(float))) );
    gpuErrchk( cudaMalloc((void**)&mat2_d, (K * M * sizeof(float))) );
    gpuErrchk( cudaMalloc((void**)&mat3_d, (N * M * sizeof(float))) );
    
    gpuErrchk( cudaMemcpy(mat1_d, mat1_h, (N * K * sizeof(float)), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(mat2_d, mat2_h, (K * M * sizeof(float)), cudaMemcpyHostToDevice) );
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(N/32.0), ceil(M/32.0), 1);
    matmulKernel<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, N, M, K);
    
    gpuErrchk( cudaMemcpy(mat3_h, mat3_d, (N * M * sizeof(float)), cudaMemcpyDeviceToHost) );
    
}
