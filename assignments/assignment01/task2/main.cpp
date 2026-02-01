#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>

using namespace std;

float** allocateMatrix(int num_rows, int num_cols);
void freeMatrix(float **mat, int num_rows);
float** addMatrix(float **mat1, float **mat2, int num_rows, int num_cols);
void readMatrix(ifstream& f, float **mat, int num_rows, int num_cols);
void printMatrix(float **mat, int num_rows, int num_cols, ostream& out);

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
        cerr << "File not open\num_rows";
        return 1;
    }
    string str;
    
    getline(ifile, str);
    int num_rows = stoi(str);
    getline(ifile, str);
    int num_cols = stoi(str);

    float **A = allocateMatrix(num_rows, num_cols);
    float **B = allocateMatrix(num_rows, num_cols);
    
    getline(ifile, str);
    readMatrix(ifile, A, num_rows, num_cols);
    readMatrix(ifile, B, num_rows, num_cols);    
    ifile.close();
    
    float **C = addMatrix(A, B, num_rows, num_cols);
    if (argc == 2)
        printMatrix(C, num_rows, num_cols, cout);
    else
    {
        ofstream ofile(argv[2]);
        printMatrix(C, num_rows, num_cols, ofile);
        ofile.close();
    }
    
    freeMatrix(A, num_rows);
    freeMatrix(B, num_rows);
    freeMatrix(C, num_rows);
}

float** allocateMatrix(int num_rows, int num_cols)
{
    float **mat = new float*[num_rows]; 
    
    for (int i = 0; i < num_rows; i++)
    {
        mat[i] = new float[num_cols]; 
    }
    
    return mat;
}

void freeMatrix(float **mat, int num_rows)
{
    for (int i = 0; i < num_rows; i++)
    {
        delete[] mat[i];
    }
    
    delete[] mat;
}

void readMatrix(ifstream& f, float **mat, int num_rows, int num_cols)
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
            mat[i][j] = stof(val);
            j++;
        }
        i++;
    }
}

void printMatrix(float **mat, int num_rows, int num_cols, ostream& out)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
             out << mat[i][j] << " ";
        }
        out << endl;
    }
}

float** addMatrix(float **mat1, float **mat2, int num_rows, int num_cols)
{
    float **mat3 = allocateMatrix(num_rows, num_cols);
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            mat3[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    
    return mat3;
}
