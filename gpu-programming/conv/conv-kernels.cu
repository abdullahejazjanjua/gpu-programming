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

__global__ void ConvKernel3(float *datain, float *dataout, int n, int m) {
    __shared__ float datain_tile[IN_DIM][IN_DIM];
    
    int row = blockIdx.y * OUT_DIM + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_DIM + threadIdx.x - FILTER_RADIUS;
    
    if ((row >= 0 && row < n) && (col >= 0 && col < m))
        datain_tile[threadIdx.y][threadIdx.x] = datain[row * m + col];
    else
        datain_tile[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    
    
    int tilerow = threadIdx.y - FILTER_RADIUS;
    int tilecol = threadIdx.x - FILTER_RADIUS;
    if (    ((row >= 0 && row < n) && (col >= 0 && col < m)) && 
        ((tilerow >= 0 && tilerow < OUT_DIM) && (tilecol >= 0 && tilecol < OUT_DIM))    ) {
        float pvalue = 0.0f;
        
        for (int fi = 0; fi < FILTER_SIZE; fi++) {
            for (int fj = 0; fj < FILTER_SIZE; fj++) {
                pvalue += (datain_tile[tilerow + fi][tilecol + fj] * FILTER_KERNEL[fi][fj]);
            }
        }  
        dataout[row * m + col] = pvalue;
    }   
}

__global__ void ConvKernel4(float *datain, float *dataout, int n, int m) {
    __shared__ float datain_tile[IN_DIM][IN_DIM];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((row >= 0 && row < n) && (col >= 0 && col < m))
        datain_tile[threadIdx.y][threadIdx.x] = datain[row * m + col];
    else
        datain_tile[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    
    
    if (row < n && col < m) {
        float pvalue = 0.0f;
        
        for (int fi = 0; fi < FILTER_SIZE; fi++) {
            for (int fj = 0; fj < FILTER_SIZE; fj++) {
                if ( ((int) threadIdx.y - FILTER_RADIUS + fi >= 0) && (threadIdx.y - FILTER_RADIUS + fi < IN_DIM)  && 
                     ((int) threadIdx.x - FILTER_RADIUS + fj >= 0) && (threadIdx.x - FILTER_RADIUS + fj < IN_DIM)
                   )
                    pvalue += (datain_tile[threadIdx.y - FILTER_RADIUS + fi][threadIdx.x - FILTER_RADIUS + fj] * FILTER_KERNEL[fi][fj]);
                // our hope is that the halo cells will most likely be in cache
                else if ( (row - FILTER_RADIUS + fi >= 0) && (row - FILTER_RADIUS + fi < n)  && 
                          (col - FILTER_RADIUS + fj >= 0) && (col - FILTER_RADIUS + fj < m)
                        )
                            pvalue += (datain[(row - FILTER_RADIUS + fi) * m + (col - FILTER_RADIUS + fj)] * FILTER_KERNEL[fi][fj]);                
            }
        }  
        dataout[row * m + col] = pvalue;
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
        case 3: {
            dim3 dimGrid(cdiv(m, OUT_DIM), cdiv(n, OUT_DIM)); // we need to launch more blocks as each block effectively 
                                                              // computes only OUT_DIM x OUT_DIM elements instead of BLOCK_SIZE x BLOCK_SIZE
            ConvKernel3<<<dimGrid, dimBlock>>>(datain_d, dataout_d, n, m);
            break;
        }
        case 4:
            ConvKernel4<<<dimGrid, dimBlock>>>(datain_d, dataout_d, n, m);
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
    printf("GPU TIME: %f microsecs\n", milliseconds * 1000);
}