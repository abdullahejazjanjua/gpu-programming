#include "hist.h"

__device__ __host__ int cdiv(int size, int block_size) {
    return (size + block_size - 1) / block_size;
}
