#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/cuda_error.h"

int main() {
    int deviceCount = 0;
    gpuErrchk( cudaGetDeviceCount(&deviceCount) );
    
    printf("Detected %d CUDA capable device(s)\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\" (Device name)\n", dev, deviceProp.name);
        printf("Major revision number:                          %d (Compute capability major version)\n", deviceProp.major);
        printf("Minor revision number:                          %d (Compute capability minor version)\n", deviceProp.minor);
        printf("Total amount of global memory:                  %.2f GB (Global memory available to the device)\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Number of multiprocessors:                      %d (Number of processing units on the device)\n", deviceProp.multiProcessorCount);
        printf("Total amount of constant memory:                %.2f KB (Constant memory available to the device)\n", (float) deviceProp.totalConstMem / (1024));
        printf("Total amount of shared memory per block:        %.2f KB (Shared memory available to each block)\n", (float) deviceProp.sharedMemPerBlock / (1024));
        printf("Total number of registers available per block:  %d (Registers available to each block)\n", deviceProp.regsPerBlock);
        printf("Warp size: %d (Number of threads in a warp)\n", deviceProp.warpSize);
        printf("Maximum number of threads per block:            %d (Max threads that can be executed in a block)\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum sizes of each dimension of a block:     %d x %d x %d (Max block dimensions)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum sizes of each dimension of a grid:      %d x %d x %d (Max grid dimensions)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Clock rate:                                     %.2f GHz (GPU core clock speed)\n", deviceProp.clockRate * 1e-6f);
        printf("Memory clock rate:                              %d MHz (Memory clock speed)\n", deviceProp.memoryClockRate / 1000);
        printf("Memory bus width:                               %d-bit (Width of the memory interface)\n", deviceProp.memoryBusWidth);
        printf("L2 cache size:                                  %d bytes (Size of the L2 cache)\n", deviceProp.l2CacheSize);

        double memClockHz = deviceProp.memoryClockRate * 1000.0;
        double busWidthBytes = deviceProp.memoryBusWidth / 8.0;
        double memBandwidth = (2.0 * memClockHz * busWidthBytes) / 1e9;
        
        int coresPerSM = 128; 
        double clockGHz = deviceProp.clockRate * 1e-6;
        double peakGFLOPS = (double)deviceProp.multiProcessorCount * coresPerSM * clockGHz * 2.0;
        
        printf("Max Global Memory Bandwidth:                    %.2f GB/s\n", memBandwidth);
        printf("Peak Compute Performance (FP32):                %.2f GFLOPS\n", peakGFLOPS);
    }

    return 0;
}