#include <iostream>
#include <vector>

#include "macros.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "USAGE: ./a.out <in-rows> <in-columns>"
                << std::endl;
        return 1;
    }
    
    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    
    std::vector<float> datain(n * m);
    std::vector<float> filter(FILTER_SIZE * FILTER_SIZE);
    std::vector<float> dataout_gpu(n * m);
    
    FillWithData(datain);
    FillWithData(filter);
    
    std::vector<float> dataout_cpu = ComputeTime(datain, filter, n, m, ConvCPU);
    std::cout << "V1: \n";
    ConvGpu(datain.data(), dataout_gpu.data(), filter.data(), n, m, 1);
    std::cout << "V2\n";
    ConvGpu(datain.data(), dataout_gpu.data(), filter.data(), n, m, 2);
    
    return 0;
}