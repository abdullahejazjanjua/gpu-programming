#include <stdio.h>
#include <stdlib.h>

__global__ void grayscaleKernel(unsigned char *data_in, unsigned char *data_out, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) 
    {
        int gray_idx = row * width + col;
        int idx = gray_idx * 3;

        unsigned char b = data_in[idx + 0];
        unsigned char g = data_in[idx + 1];
        unsigned char r = data_in[idx + 2];

        data_out[gray_idx] = (unsigned char)(0.216 * r + 0.7152 * g + 0.0722 * b);
  }
}


void grayscale(unsigned char *data_inh, unsigned char *data_outh, int height, int width) 
{
    unsigned char *data_ind, *data_outd;
    cudaError_t err;
    
    err = cudaMalloc((void **)&data_ind, (height * width * 3 * sizeof(unsigned char)));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&data_outd, (height * width * sizeof(unsigned char)));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(data_ind, data_inh, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(width/32.0), ceil(height/32.0), 1);
    grayscaleKernel<<<dimGrid, dimBlock>>>(data_ind, data_outd, height, width);
    
    err = cudaMemcpy(data_outh, data_outd, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
    
}