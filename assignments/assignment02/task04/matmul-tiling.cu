#include "../utils/cuda_error.h"

#define TILESIZE 32

// mat1: N, K   mat2: K, M mat3: N, M
__global__ void matmulTilingKernel(float *mat1, float *mat2, float *mat3,  int N, int M, int K) {
    __shared__ float Mds[TILESIZE][TILESIZE];
    __shared__ float Nds[TILESIZE][TILESIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // N
    int col = blockIdx.x * blockDim.x + threadIdx.x; // M
    
    float pvalue = 0.0f;
    for (int phase = 0; phase < ceil((float) K/TILESIZE); phase++) {
        
        int mat1_col = (phase * TILESIZE + threadIdx.x);
        int mat2_row = (phase * TILESIZE + threadIdx.y);
        
        if (row < N && mat1_col < K) {
            Mds[threadIdx.y][threadIdx.x] = mat1[row * K + mat1_col];
        }
        else Mds[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (mat2_row < K && col < M) {
            Nds[threadIdx.y][threadIdx.x] = mat2[mat2_row * M + col];
        }
        else Nds[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        
        for (int tile_i = 0; tile_i < TILESIZE; tile_i++) {
            pvalue += (Mds[threadIdx.y][tile_i] * Nds[tile_i][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < N && col < M) 
        mat3[row * M + col] = pvalue;
}

void mulMatrixGPUTiling(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *mat1_d, *mat2_d, *mat3_d;
    gpuErrchk( cudaMalloc((void**)&mat1_d, ((size_t) N * K * sizeof(float))) );
    gpuErrchk( cudaMalloc((void**)&mat2_d, ((size_t) K * M * sizeof(float))) );
    gpuErrchk( cudaMalloc((void**)&mat3_d, ((size_t) N * M * sizeof(float))) );
    
    cudaEventRecord(start);
    gpuErrchk( cudaMemcpy(mat1_d, mat1_h, ((size_t) N * K * sizeof(float)), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(mat2_d, mat2_h, ((size_t) K * M * sizeof(float)), cudaMemcpyHostToDevice) );

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((M + 31) / 32, (N + 31) / 32, 1);
    matmulTilingKernel<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, N, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    gpuErrchk( cudaMemcpy(mat3_h, mat3_d, ((size_t) N * M * sizeof(float)), cudaMemcpyDeviceToHost) );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Tiling TIME: %f microsecs\n", milliseconds * 1000);
    
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(mat3_d);
    
}