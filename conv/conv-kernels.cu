#include "common.h"
#include <cuda.h>
#include <cuda_device_runtime_api.h>

#define FILTER_SIZE(r) ((2 * r) + 1)

__global__ void ConvKernel1(float *datain, float *dataout, float *filter, int n,
                            int m, int filter_radius) {
  int outrow = blockIdx.y * blockDim.y + threadIdx.y;
  int outcol = blockIdx.x * blockDim.x + threadIdx.x;

  if ((outrow >= 0 && outrow < n) && (outcol >= 0 && outcol < m)) {
    float pvalue = 0.0f;

    for (int fi = 0; fi < FILTER_SIZE(filter_radius); fi++) {
      for (int fj = 0; fj < FILTER_SIZE(filter_radius); fj++) {
        int inrow = outrow + fi - filter_radius;
        int incol = outcol + fj - filter_radius;

        if ((inrow >= 0 && inrow < n) && (incol >= 0 && incol < m))
          pvalue += (datain[inrow * m + incol] *
                     filter[fi * FILTER_SIZE(filter_radius) + fj]);
      }
    }
    
    dataout[outrow * m + outcol] = pvalue;
  }
}

void conv1(float *datain, float *dataout, float *filter, int n, int m,
           int filter_radius) {
  float *datain_d, *dataout_d, *filter_d;
  int filter_size = FILTER_SIZE(filter_radius);

  CHECK_CUDA(cudaMalloc((void **)&datain_d, (n * m * sizeof(float))));
  CHECK_CUDA(cudaMalloc((void **)&dataout_d, (n * m * sizeof(float))));
  CHECK_CUDA(cudaMalloc((void **)&filter_d,
                        (filter_size * filter_size * sizeof(float))));

  CHECK_CUDA(cudaMemcpy(datain_d, datain, (n * m * sizeof(float)),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(filter_d, filter,
                        (filter_size * filter_size * sizeof(float)),
                        cudaMemcpyHostToDevice));

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((m + 31) / 32, (n + 31) / 32, 1);
  ConvKernel1<<<dimGrid, dimBlock>>>(datain_d, dataout_d, filter_d, n, m,
                                     filter_radius);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(dataout, dataout_d, (n * m * sizeof(float)),
                        cudaMemcpyDeviceToHost));

  cudaFree(datain_d);
  cudaFree(dataout_d);
  cudaFree(filter_d);
}