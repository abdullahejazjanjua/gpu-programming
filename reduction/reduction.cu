#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

__global__ void reductionv1(int *in, int *out, int len) 
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0)
        {
            if (idx < len && (idx + stride < len))
                in[idx] = in[idx] + in[idx + stride];
        }
         __syncthreads();
    }

    if (idx == 0) 
    {
        *out = in[0];
    }
}


__global__ void reductionv2(int *in, int *out, int len) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride) 
        {
            if (idx < len && (idx + stride) < len)
                in[idx] = in[idx] + in[idx + stride];
        }
        __syncthreads();
    }
    if (idx == 0) 
        *out = in[0];
}

__global__ void reductionv3(int *in, int *out, int len) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int in_s[BLOCK_DIM];
    
    int sum = 0;
    if (idx < len) sum += in[idx];
    if ((idx + BLOCK_DIM) < len) sum += in[idx + BLOCK_DIM];
    in_s[threadIdx.x] = sum;

    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            in_s[threadIdx.x] += in_s[threadIdx.x + stride];
    }

    if (idx == 0) 
        *out = in_s[0];
}

__global__ void reductionv4(int *in, int *out, int len) 
{
    int segment = blockIdx.x * (2 * blockDim.x);
    int idx = segment + threadIdx.x;

    __shared__ int in_s[BLOCK_DIM];
    
    int sum = 0;
    if (idx < len) sum += in[idx];
    if ((idx + BLOCK_DIM) < len) sum += in[idx + BLOCK_DIM];
    in_s[threadIdx.x] = sum;

    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            in_s[threadIdx.x] += in_s[threadIdx.x + stride];
    }
    
    if (threadIdx.x == 0) 
        atomicAdd(out, in_s[0]);
}

__global__ void reductionv5(int *in, int *out, int len) 
{
    int segment = COARSE_FACTOR * 2 * blockIdx.x * blockDim.x;
    int idx = segment + threadIdx.x;
    
    __shared__ int in_s[BLOCK_DIM];
    int sum = 0;
    for (int i = 0; i < COARSE_FACTOR * 2; i++)
    {
        if ((idx + i * BLOCK_DIM) < len)
             sum += in[idx + i * BLOCK_DIM];
    }

    in_s[threadIdx.x] = sum;
    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            in_s[threadIdx.x] += in_s[threadIdx.x + stride];
    }

    if (threadIdx.x == 0) 
        atomicAdd(out, in_s[0]);
}


// Q3 from chap 10 of PMPP
__global__ void reductionv6(int *in, int *out, int len) 
{
    int idx = threadIdx.x + blockDim.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (blockDim.x - threadIdx.x <= stride)
            in[idx] = in[idx] + in[idx - stride];
        __syncthreads();
    }
    if (threadIdx.x == blockDim.x - 1) 
    {
        *out = in[idx];
    }
}