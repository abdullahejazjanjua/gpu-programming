#include "macros.h"
#include <cuda.h>
#include <cuda_runtime.h>


__constant__ float FILTER_KERNEL[FILTER_SIZE][FILTER_SIZE];

__global__ void ConvKernel1(float *datain, float *dataout, float *filter, int n, int m) {
    int outrow = blockIdx.y * blockDim.y + threadIdx.y;
    int outcol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outrow < n && outcol < m) {
        float pvalue = 0.0f;
    
        for (int fi = 0; fi < FILTER_SIZE; fi++) {
            for (int fj = 0; fj < FILTER_SIZE; fj++) {
                int inrow = outrow + fi - FILTER_RADIUS;
                int incol = outcol + fj - FILTER_RADIUS;
        
                if ((inrow >= 0 && inrow < n) && (incol >= 0 && incol < m))
                    pvalue += (datain[inrow * m + incol] * filter[fi * FILTER_SIZE + fj]);
            }
        }
        
        dataout[outrow * m + outcol] = pvalue;
    }
}

__global__ void ConvKernel2(float *datain, float *dataout, int n, int m) {
    int outrow = blockIdx.y * blockDim.y + threadIdx.y;
    int outcol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outrow < n && outcol < m) {
        float pvalue = 0.0f;
    
        for (int fi = 0; fi < FILTER_SIZE; fi++) {
            for (int fj = 0; fj < FILTER_SIZE; fj++) {
                int inrow = outrow + fi - FILTER_RADIUS;
                int incol = outcol + fj - FILTER_RADIUS;
        
                if ((inrow >= 0 && inrow < n) && (incol >= 0 && incol < m))
                    pvalue += (datain[inrow * m + incol] * FILTER_KERNEL[fi][fj]);
            }
        }
        
        dataout[outrow * m + outcol] = pvalue;
    }
}

int cdiv(int size, int block_size) {
    return (size + block_size - 1) / block_size;
}

void ConvGpu(float *datain, float *dataout, float *filter, int n, int m, int conv_version) {
  
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *datain_d, *dataout_d, *filter_d;
    
    CHECK_CUDA(cudaMalloc((void **)&datain_d, (n * m * sizeof(float))));
    CHECK_CUDA(cudaMalloc((void **)&dataout_d, (n * m * sizeof(float))));
    if (conv_version == 1)
        CHECK_CUDA(cudaMalloc((void **)&filter_d, (FILTER_SIZE * FILTER_SIZE * sizeof(float))));
    
    CHECK_CUDA(cudaMemcpy(datain_d, datain, (n * m * sizeof(float)), cudaMemcpyHostToDevice));
    if (conv_version == 1)
        CHECK_CUDA(cudaMemcpy(filter_d, filter, (FILTER_SIZE * FILTER_SIZE * sizeof(float)), cudaMemcpyHostToDevice));
    else
        CHECK_CUDA(cudaMemcpyToSymbol(FILTER_KERNEL, filter, (FILTER_SIZE * FILTER_SIZE * sizeof(float))));
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(cdiv(m, BLOCK_SIZE), cdiv(n, BLOCK_SIZE));
    
    cudaEventRecord(start);
    switch (conv_version) {
        case 1: 
            ConvKernel1<<<dimGrid, dimBlock>>>(datain_d, dataout_d, filter_d, n, m);
            break;
        case 2:
            ConvKernel2<<<dimGrid, dimBlock>>>(datain_d, dataout_d, n, m);
            break;
        default:
            fprintf(stderr, "No such kernel version found\n");
            exit(EXIT_FAILURE);
    }
    
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaMemcpy(dataout, dataout_d, (n * m * sizeof(float)), cudaMemcpyDeviceToHost));
    
    cudaFree(datain_d);
    cudaFree(dataout_d);
    if (conv_version == 1)
        cudaFree(filter_d);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    GPU TIME: %f microsecs\n", milliseconds * 1000);
}