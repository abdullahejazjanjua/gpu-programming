#include "utils.h"
#include <iostream>

#define FILTER_SIZE(r) ((2 * r) + 1)

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "USAGE: ./a.out <in-rows> <in-columns> <filter-radius>"
              << std::endl;
    return 1;
  }

  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int filter_radius = std::stoi(argv[3]);

  std::vector<float> datain(n * m);
  std::vector<float> filter(FILTER_SIZE(filter_radius) *
                            FILTER_SIZE(filter_radius));
  std::vector<float> dataout_gpu(n * m);
  
  FillWithData(datain);
  FillWithData(filter);

  std::vector<float> dataout_cpu =
      ConvCPU(datain, filter, n, m, filter_radius);

  conv1(datain.data(), dataout_gpu.data(), filter.data(), n, m, filter_radius);
  CheckCorrectness(dataout_cpu, dataout_gpu, n, m);

  return 0;
}