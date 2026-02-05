#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "USAGE: ./a.out <num_rows> <num_cols>\n";
        return 1;
    }
    int num_rows = stoi(argv[1]);
    int num_cols = stoi(argv[2]);
    
    float *mat = new float[num_rows * num_cols];
    if (mat == nullptr)
    {
        cerr << "Allocation failed\n";
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            mat[i * num_cols + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    
    ofstream ofile("input.txt");
    ofile << num_rows << "\n";
    ofile << num_cols << "\n";
    
    ofile << "\n";
    
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            if (j == num_cols - 1)
                ofile << mat[i * num_cols + j];
            else
                ofile << mat[i * num_cols + j] << " ";
        }
        ofile << "\n";
    }
    
    ofile << "\n";
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            if (j == num_cols - 1)
                ofile << mat[i * num_cols + j];
            else
                ofile << mat[i * num_cols + j] << " ";
        }
        ofile << "\n";
    }
    
}