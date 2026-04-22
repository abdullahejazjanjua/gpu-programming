#include "hist.h"

#define COARSE_FACTOR 4

__global__ void histogramv1(char *data, int *hist, ull len) {
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        int pos = data[idx] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(hist[pos/4]), 1);
        }
    }
}

__global__ void histogramv2(char *data, int *hist, ull len) {
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int hist_s[MAX_INTERVALS];
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) hist_s[i] = 0;

    __syncthreads();
    if (idx < len) {
        int pos = data[idx] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(hist_s[pos/4]), 1);
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) {
        int val = hist_s[i];
        if (val > 0) 
            atomicAdd(&hist[i], val);
    }
}

__global__ void histogramv3(char *data, int *hist, ull len) {
    __shared__ int hist_s[MAX_INTERVALS];
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) hist_s[i] = 0;

    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (int i = tid*COARSE_FACTOR; i < min(COARSE_FACTOR * (tid+1), len); i++) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) 
            atomicAdd(&(hist_s[pos/4]), 1);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) {
        int val = hist_s[i];
        if (val > 0) 
            atomicAdd(&hist[i], val);
    }
}


__global__ void histogramv4(char *data, int *hist, ull len) {
    __shared__ int hist_s[MAX_INTERVALS];
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) hist_s[i] = 0;

    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (int i = tid; i < len; i+=(gridDim.x * blockDim.x)) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) 
            atomicAdd(&(hist_s[pos/4]), 1);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) {
        int val = hist_s[i];
        if (val > 0) 
            atomicAdd(&hist[i], val);
    }
}

__global__ void histogramv5(char *data, int *hist, ull len) {
    __shared__ int hist_s[MAX_INTERVALS];
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) hist_s[i] = 0;

    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    int acc = 0;
    int prevBinIdx = -1;
    for (int i = tid; i < len; i+=(gridDim.x * blockDim.x)) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            int bin = pos/4;
            if (bin == prevBinIdx)
                acc++;
            else {
                if (acc > 0)
                    atomicAdd(&(hist_s[prevBinIdx]), acc);
                acc = 1;
                prevBinIdx = bin;
            }
        }
    }

    if (acc > 0)
        atomicAdd(&(hist_s[prevBinIdx]), acc);

    __syncthreads();
    for (int i = threadIdx.x; i < MAX_INTERVALS; i += blockDim.x) {
        int val = hist_s[i];
        if (val > 0) 
            atomicAdd(&hist[i], val);
    }
}



void histogram_gpu(char *data, int *hist, ull len, int version) {
    char *d_data; int *d_hist;
    CUDA_CHECK( cudaMalloc((void**)&d_data, len * sizeof(char)) );
    CUDA_CHECK( cudaMalloc((void**)&d_hist, MAX_INTERVALS * sizeof(int)) );

    CUDA_CHECK( cudaMemcpy(d_data, data, len * sizeof(char), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_hist, hist, MAX_INTERVALS * sizeof(int), cudaMemcpyHostToDevice) );

    dim3 dimBlock(1024, 1);
    dim3 dimGrid(cdiv(len, 1024), 1, 1);
    if (version >= 3)
        dim3 dimGrid(cdiv(len, 1024)/COARSE_FACTOR, 1, 1);

    cudaEvent_t start, stop;
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );
    CUDA_CHECK( cudaEventRecord(start) );

    switch(version) {
        case 1: histogramv1<<<dimGrid, dimBlock>>>(d_data, d_hist, len); break;
        case 2: histogramv2<<<dimGrid, dimBlock>>>(d_data, d_hist, len); break;
        case 3: histogramv3<<<dimGrid, dimBlock>>>(d_data, d_hist, len); break;
        case 4: histogramv4<<<dimGrid, dimBlock>>>(d_data, d_hist, len); break;
        case 5: histogramv5<<<dimGrid, dimBlock>>>(d_data, d_hist, len); break;
        default: printf("Invalid version\n"); return;
    }

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaEventRecord(stop) );
    CUDA_CHECK( cudaEventSynchronize(stop) );

    float ms;
    CUDA_CHECK( cudaEventElapsedTime(&ms, start, stop) );
    printf("Kernel time (v%d): %.3f ms\n", version, ms);

    CUDA_CHECK( cudaMemcpy(hist, d_hist, MAX_INTERVALS * sizeof(int), cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_data) );
    CUDA_CHECK( cudaFree(d_hist) );
    
    CUDA_CHECK( cudaEventDestroy(start) );
    CUDA_CHECK( cudaEventDestroy(stop) );
}
