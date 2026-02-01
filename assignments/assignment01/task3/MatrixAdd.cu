__global__ void matrixAddKernel(float **mat1, float **mat2, float **mat3, int num_rows, int num_cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    
    if (i < num_rows && j < num_cols)
    {
        mat3[i][j] = mat1[i][j] + mat2[i][j];
    }
}


void matrixCopy(float **matdest, float **matsrc, int num_rows, int num_cols, cudaMemcpyKind direction)
{
    for (int i = 0; i < num_rows; i++)
    {
        cudaMemcpy()
        for (int j = 0; j < num_cols; j++)
        {
            matdest[i * num_cols + j] = matsrc[i][j];
        }
    }
        
}

float** matrixAllocate(int num_rows, int num_cols)
{
    float **mat;
    cudaMalloc((void **)&mat, (num_rows * num_cols * sizeof(float*)));
    return mat;
}

void matrixAdd(float **mat1_h, float **mat2_h, float **mat3_h, int num_rows, int num_cols)
{
    float *mat1_d = matrixAllocate(num_rows, num_cols);
    float *mat2_d = matrixAllocate(num_rows, num_cols);
    float *mat3_d = matrixAllocate(num_rows, num_cols);
    
    matrixCopy(mat1_d, mat1_h, num_rows, num_cols, cudaMemcpyHostToDevice);
    matrixCopy(mat2_d, mat2_h, num_rows, num_cols, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16, 1); // x (columns), y (rows), z (depth)
    dim3 gridDim(ceil(num_cols/16), ceil(num_rows/16), 1); // x (columns), y (rows), z (depth)
    matrixAddKernel<<<gridDim, blockDim>>>(mat1_d, mat2_d, mat3_d, num_rows, num_cols);
    
    matrixCopy(mat3_h, mat3_d, num_rows, num_cols, cudaMemcpyDeviceToHost);
    
}