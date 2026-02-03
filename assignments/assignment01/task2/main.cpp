#include <cstdlib>
#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>

#include <chrono>

using namespace std;

float* allocateMatrix(int num_rows, int num_cols);
void freeMatrix(float *mat, int num_rows);
float* addMatrix(float *mat1, float *mat2, int num_rows, int num_cols);
void readMatrix(ifstream& f, float *mat, int num_rows, int num_cols);
void printMatrix(float *mat, int num_rows, int num_cols, ostream& out);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "USAGE: ./a.out <input-file>.txt [<output-file>.txt]\num_rows";
        return 1;
    }
    
    ifstream ifile(argv[1]);
    if (!ifile.is_open())
    {
        cerr << "File not open\n";
        return 1;
    }
    string str;
    
    getline(ifile, str);
    int num_rows = stoi(str);
    getline(ifile, str);
    int num_cols = stoi(str);

    float *A = allocateMatrix(num_rows, num_cols);
    float *B = allocateMatrix(num_rows, num_cols);
    
    getline(ifile, str);
    readMatrix(ifile, A, num_rows, num_cols);
    readMatrix(ifile, B, num_rows, num_cols);    
    ifile.close();
    
    auto begin = chrono::high_resolution_clock::now();
    float *C = addMatrix(A, B, num_rows, num_cols);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - begin);
    
    if (argc == 2)
    {
        cout << "Time taken: " << duration.count() << "mus" << endl;
        printMatrix(C, num_rows, num_cols, cout);
    }
    else
    {
        ofstream ofile(argv[2]);
        ofile << num_rows << "\n";
        ofile << num_cols << "\n\n";
        ofile << "Time taken: " << duration.count() << "mus" << "\n\n";
        printMatrix(C, num_rows, num_cols, ofile);
        ofile.close();
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
}

float* allocateMatrix(int num_rows, int num_cols)
{
    float *mat = new float[num_rows * num_cols]; 
    if (mat == nullptr)
    {
        cerr << "Couldn't allocate memory\n";
        exit(EXIT_FAILURE);
    }
    return mat;
}

void readMatrix(ifstream& f, float *mat, int num_rows, int num_cols)
{
    string str;
    int i = 0;
    while (getline(f, str) && i != num_rows)
    {
        string val;
        int j = 0;
        stringstream ss(str);
        while (getline(ss, val, ' ') && j < num_cols)
        {
            mat[i * num_cols + j] = stof(val);
            j++;
        }
        i++;
    }
}

void printMatrix(float *mat, int num_rows, int num_cols, ostream& out)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
             out << mat[i * num_cols + j] << " ";
        }
        out << endl;
    }
}

float* addMatrix(float *mat1, float *mat2, int num_rows, int num_cols)
{
    float *mat3 = allocateMatrix(num_rows, num_cols);
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            mat3[i * num_cols + j] = mat1[i * num_cols + j] + mat2[i * num_cols + j];
        }
    }
    
    return mat3;
}
