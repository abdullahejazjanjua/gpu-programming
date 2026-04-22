#include <iostream>
#include <cstring>
#include <vector>
#include <climits>
#include "hist.h"


using std::cout;
using std::vector;

void histogram_cpu(const vector<char> data, int *hist, ull len) {
    for (ull i = 0; i < len; i++) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26)
            hist[pos/4]++;
    }
}


int main() {
    int hist_cpu[MAX_INTERVALS];
    int hist_gpu[MAX_INTERVALS];
    
    vector<char> data;
    for (ull i = 0; i < MAX_DATA; i++) {
        char c = 'a' + (rand() % 26);
        data.push_back(c);
    }
    
    for (int i = 0; i < MAX_INTERVALS; i++) {
        hist_cpu[i] = 0;
    }

    histogram_cpu(data, hist_cpu, MAX_DATA);
    for (int i = 1; i <= NUM_VERSIONS; i++) {
        printf("\nProcessing Version: %d\n", i);
        for (int i = 0; i < MAX_INTERVALS; i++) 
            hist_gpu[i] = 0;
        histogram_gpu(data.data(), hist_gpu, MAX_DATA, i);
    
    
        int err = 0;
        for (int i = 0; i < MAX_INTERVALS; i++) {
            err += abs(hist_cpu[i] - hist_gpu[i]);
        }

        cout << "Difference b/w CPU and GPU: " << err << "\n";
        cout << "GPU: ";
        for (int i = 0; i < MAX_INTERVALS; i++) cout << hist_gpu[i] << " ";
        cout << "\n";
        cout << "CPU: ";
        for (int i = 0; i < MAX_INTERVALS; i++) cout << hist_cpu[i] << " ";
        cout << "\n";
    }

    return 0;
}