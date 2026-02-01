#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>

using namespace std;

int** allocateMatrix(int N);
void freeMatrix(int **mat, int N);

void readMatrix(ifstream& f, int **mat, int N);
void printMatrix(int **mat, int N, ostream& out);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "USAGE: ./a.out <input-file>.txt [<output-file>.txt]\n";
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
    int N = stoi(str);

    int **A = allocateMatrix(N);
    int **B = allocateMatrix(N);
    
    getline(ifile, str);
    readMatrix(ifile, A, N);
    readMatrix(ifile, B, N);
    
    
    ifile.close();
    
    int **C = addMatrix(A, B, N);
    
    if (argc == 2)
        printMatrix(C, N, cout);
    else
    {
        ofstream ofile(argv[2]);
        printMatrix(C, N, ofile);
        ofile.close();
    }
    
    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C, N);
}

int** allocateMatrix(int N)
{
    int **mat = new int*[N]; 
    
    for (int i = 0; i < N; i++)
    {
        mat[i] = new int[N]; 
    }
    
    return mat;
}

void freeMatrix(int **mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        delete[] mat[i];
    }
    
    delete[] mat;
}

void readMatrix(ifstream& f, int **mat, int N)
{
    string str;
    int i = 0;
    while (getline(f, str) && i != N)
    {
        string val;
        int j = 0;
        stringstream ss(str);
        while (getline(ss, val, ' ') && j < N)
        {
            mat[i][j] = stoi(val);
            j++;
        }
        i++;
    }
}

void printMatrix(int **mat, int N, ostream& out)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
             out << mat[i][j] << " ";
        }
        out << endl;
    }
}
