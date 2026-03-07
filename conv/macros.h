#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(code)                                                       \
  do {                                                                         \
    if ((code) != cudaSuccess) {                                               \
      std::cerr << "GPU ERROR in " << __FILE__ << ":" << __LINE__              \
                << " := " << cudaGetErrorString(code) << "\n";                 \
      exit(code);                                                              \
    }                                                                          \
  } while (0)


#define FILTER_RADIUS 3
#define FILTER_SIZE ((2 * (FILTER_RADIUS)) + 1)

#define IN_DIM 32
#define BLOCK_SIZE (IN_DIM)

#define OUT_DIM ((IN_DIM) - 2*(FILTER_RADIUS))
