#include <cuda_device_runtime_api.h>
#include <stdio.h>

#define TILE_SIZE 32
#define COARSE_FACTOR 4

#define CHECK_CUDA(code)                                                       \
  do {                                                                         \
    if ((code) != cudaSuccess) {                                               \
      fprintf(stderr, "GPU ERROR %s: line %d :=  %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(code));                                       \
      exit(code);                                                              \
    }                                                                          \
  } while (0)

__global__ void MatmulKernelColMajor(float *mat1, float *mat2, float *mat3,
                                     int n, int m, int k) {

  __shared__ float mat1_s[TILE_SIZE][TILE_SIZE];
  __shared__ float mat2_s[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float pvalue = 0.0f;
  for (int phase = 0; phase < ceil((float)k / TILE_SIZE); phase++) {

    int mat1_col = phase * TILE_SIZE + threadIdx.x;
    int mat2_row = phase * TILE_SIZE + threadIdx.y;

    if (row < n && mat1_col < k)
      mat1_s[threadIdx.y][threadIdx.x] = mat1[row * k + mat1_col];
    else
      mat1_s[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < m &&
        mat2_row < k) // as col is acting as row, its must be within row range
                      // likewise for mat2_row which is acting as column
      mat2_s[threadIdx.y][threadIdx.x] = mat2[col * k + mat2_row];
    else
      mat2_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      pvalue += (mat1_s[threadIdx.y][i] * mat2_s[i][threadIdx.x]);
    }
    __syncthreads();
  }

  if (row < n && col < m) {
    mat3[row * m + col] = pvalue;
  }
}

__global__ void MatmulKernelRowMajor(float *mat1, float *mat2, float *mat3,
                                     int n, int m, int k) {
  __shared__ float Mds[TILE_SIZE][TILE_SIZE];
  __shared__ float Nds[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y; // n
  int col = blockIdx.x * blockDim.x + threadIdx.x; // m

  float pvalue = 0.0f;
  for (int phase = 0; phase < ceil((float)k / TILE_SIZE); phase++) {

    int mat1_col = (phase * TILE_SIZE + threadIdx.x);
    int mat2_row = (phase * TILE_SIZE + threadIdx.y);

    if (row < n && mat1_col < k) {
      Mds[threadIdx.y][threadIdx.x] = mat1[row * k + mat1_col];
    } else
      Mds[threadIdx.y][threadIdx.x] = 0.0f;

    if (mat2_row < k && col < m) {
      Nds[threadIdx.y][threadIdx.x] = mat2[mat2_row * m + col];
    } else
      Nds[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    for (int tile_i = 0; tile_i < TILE_SIZE; tile_i++) {
      pvalue += (Mds[threadIdx.y][tile_i] * Nds[tile_i][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < n && col < m)
    mat3[row * m + col] = pvalue;
}

__global__ void MatmulKernelRowMajorCoarsing(float *mat1, float *mat2,
                                             float *mat3, int n, int m, int k) {
  __shared__ float Mds[TILE_SIZE][TILE_SIZE];
  __shared__ float Nds[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;                       // n
  int col_start = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x; // m

  float pvalue[COARSE_FACTOR];
  for (int i = 0; i < COARSE_FACTOR; i++)
    pvalue[i] = 0.0f;

  for (int phase = 0; phase < ceil((float)k / TILE_SIZE); phase++) {

    int mat1_col = (phase * TILE_SIZE + threadIdx.x);
    int mat2_row = (phase * TILE_SIZE + threadIdx.y);

    if (row < n && mat1_col < k) {
      Mds[threadIdx.y][threadIdx.x] = mat1[row * k + mat1_col];
    } else
      Mds[threadIdx.y][threadIdx.x] = 0.0f;

    for (int c = 0; c < COARSE_FACTOR; c++) {
      int col = col_start + c * TILE_SIZE;

      if (mat2_row < k && col < m) {
        Nds[threadIdx.y][threadIdx.x] = mat2[mat2_row * m + col];
      } else
        Nds[threadIdx.y][threadIdx.x] = 0.0f;
      __syncthreads();

      for (int tile_i = 0; tile_i < TILE_SIZE; tile_i++) {
        pvalue[c] += (Mds[threadIdx.y][tile_i] * Nds[tile_i][threadIdx.x]);
      }
      __syncthreads();
    }
  }

  for (int c = 0; c < COARSE_FACTOR; c++) {
    int col = col_start + c * TILE_SIZE;
    if (row < n && col < m) {
        mat3[row * m + col] = pvalue[c];
    }
  }
}

void MatMulGpu(float *mat1_h, float *mat2_h, float *mat3_h, int n, int m, int k,
               bool is_row_major = false, bool thread_coarsing = false) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *mat1_d, *mat2_d, *mat3_d;
  CHECK_CUDA(cudaMalloc((void **)&mat1_d, ((size_t)n * k * sizeof(float))));
  CHECK_CUDA(cudaMalloc((void **)&mat2_d, ((size_t)m * k * sizeof(float))));
  CHECK_CUDA(cudaMalloc((void **)&mat3_d, ((size_t)n * m * sizeof(float))));

  cudaEventRecord(start);
  CHECK_CUDA(cudaMemcpy(mat1_d, mat1_h, ((size_t)n * k * sizeof(float)),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(mat2_d, mat2_h, ((size_t)m * k * sizeof(float)),
                        cudaMemcpyHostToDevice));

  dim3 dimBlock(TILE_SIZE, TILE_SIZE);

  if (thread_coarsing) {
    dim3 dimGrid((m + (TILE_SIZE * COARSE_FACTOR) - 1) / (TILE_SIZE * COARSE_FACTOR), (n + TILE_SIZE - 1) / TILE_SIZE);
    MatmulKernelRowMajorCoarsing<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, n, m, k);
  }
  else if (is_row_major) {
    dim3 dimGrid((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    MatmulKernelRowMajor<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, n, m, k);
  }
  else {
    dim3 dimGrid((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    MatmulKernelColMajor<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, n, m, k);
  }

  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(mat3_h, mat3_d, ((size_t)n * m * sizeof(float)),
                        cudaMemcpyDeviceToHost));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU TIME: %f microsecs\n", milliseconds * 1000);

  cudaFree(mat1_d);
  cudaFree(mat2_d);
  cudaFree(mat3_d);
}