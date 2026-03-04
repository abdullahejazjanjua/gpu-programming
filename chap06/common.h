#ifndef COMMON_H
#define COMMON_H

void MatMulGpu(float *mat1_h, float *mat2_h, float *mat3_h, int n,
                        int m, int K, bool is_row_major=false, bool thread_coarsing=false);

#endif