#ifndef HIST_H
#define HIST_H

#include <cuda_runtime.h>
#include <iostream>

#define MAX_INTERVALS 7
#define MAX_DATA 100000000UL
#define NUM_VERSIONS 5

typedef unsigned long long ull;


#define CUDA_CHECK(code)                                                       \
  do {                                                                         \
    if ((code) != cudaSuccess) {                                               \
      std::cerr << "GPU ERROR in " << __FILE__ << ":" << __LINE__              \
                << " := " << cudaGetErrorString(code) << "\n";                 \
      exit(code);                                                              \
    }                                                                          \
  } while (0)


int cdiv(int size, int block_size);
void histogram_gpu(char *data, int *hist, ull len, int version);

#endif