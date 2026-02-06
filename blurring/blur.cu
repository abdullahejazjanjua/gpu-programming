#include <stdio.h>
#include <stdlib.h>

__global__ void blurKernel(unsigned char *data_in, unsigned char *data_out, int height, int width, int blurstep) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        
        int r = 0, g = 0, b = 0; 
        int pixels = 0;
        
        for (int i =-blurstep; i < blurstep + 1; i++) {
            for (int j =-blurstep; j < blurstep + 1; j++) {
                int curRow = row + i;
                int curCol = col + j;
                
                if ((curRow >= 0 && curRow < height) && (curCol >= 0 && curCol < width)) {
                    int curIdx = (curRow * width + curCol) * 3;
                    b += data_in[curIdx + 0];
                    g += data_in[curIdx + 1];
                    r += data_in[curIdx + 2];
                    pixels++;
                }
            }
        }
        data_out[idx + 0] = (unsigned char) (b/pixels);
        data_out[idx + 1] = (unsigned char) (g/pixels);
        data_out[idx + 2] = (unsigned char) (r/pixels);
    }
}


void blur(unsigned char *data_inh, unsigned char *data_outh, int height, int width, int blurstep) 
{
    unsigned char *data_ind, *data_outd;
    cudaError_t err;
    
    err = cudaMalloc((void **)&data_ind, (height * width * 3 * sizeof(unsigned char)));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&data_outd, (height * width * 3 * sizeof(unsigned char)));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory in device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(data_ind, data_inh, (height * width * 3 * sizeof(unsigned char)), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(width/32.0), ceil(height/32.0), 1);
    blurKernel<<<dimGrid, dimBlock>>>(data_ind, data_outd, height, width, blurstep);
    if (errSync != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed (error code %s)!\n", cudaGetErrorString(errSync));
            exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(data_outh, data_outd, (height * width * 3 * sizeof(unsigned char)), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
    
    cudaFree(data_ind);
    cudaFree(data_outd);
    
}