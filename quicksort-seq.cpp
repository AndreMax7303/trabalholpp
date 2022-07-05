#include <mpi.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <chrono>
using namespace std::chrono;
using namespace std;


int getDatasetSize(string filename);
void setDataset(string filename, double *pDataset);
void ProcessInitialization(double **pProcData, int *ProcDataSize);
void ProcessTermination(double* pProcData, int ProcDataSize);
void swap(double *pProcData, int pos1, int pos2);
void sequentialQuickSort(double *pProcData, int pos1, int pos2);
void PrintData(double *pData, int dataSize);


int ProcRank; // Rank of current process
int ProcNum;  // Number of processes
double start;
double stop;

int main(int argc, char *argv[])
{
    double *pProcData; // Data block for the process
    int ProcDataSize;  // Data block size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    if (ProcRank == 0) {
        start = MPI_Wtime();
    }
    // Data Initialization
    ProcessInitialization(&pProcData, &ProcDataSize);
    // sorting
    sequentialQuickSort(pProcData, 0, ProcDataSize);
    // The termination of process computations
    // ProcessTermination(pProcData, ProcDataSize);

    if (ProcRank == 0) {
        stop = MPI_Wtime();
        cout << "Duração: " << stop - start << endl;
    }
    MPI_Finalize();
}

void PrintData(double *pData, int dataSize){
    string temp = "Proc " + to_string(ProcRank)+ ":";
    for (int i = 0; i < dataSize; i++){
        temp += to_string(*(pData+i)) + " ";
    }
    cout << temp << endl;
}

int getDatasetSize(string filename)
{
    ifstream myFile;
    string temp;

    myFile.open(filename);

    int count = 0;
    while (myFile >> temp){
        count++;
    }
    myFile.close();
    return count;
}

void setDataset(string filename, double *pDataset)
{
    ifstream myFile;
    string temp;
    myFile.open(filename);
    int count = 0;
    while(myFile >> temp) {
        pDataset[count] = stod(temp);
        count++;
    }
    
    myFile.close();
}

void ProcessInitialization(double **pProcData, int *ProcDataSize) {
    string filename = "dataset1000000.txt";
    int datasetSize;
    double *dataset;
    
    datasetSize = getDatasetSize(filename);
    cout << "Tamanho do dataset: " << datasetSize << endl;
    dataset = new double[datasetSize];
    setDataset(filename, dataset);
    *ProcDataSize = datasetSize/ProcNum;
    *pProcData = dataset;
    
}

void swap(double *pProcData, int pos1, int pos2){
    double temp = *(pProcData + pos1);
    *(pProcData + pos1) = *(pProcData + pos2);
    *(pProcData + pos2) = temp;
}

void sequentialQuickSort(double *pProcData, int pos1, int pos2){
    if (pos1 < pos2){
        double pivot = *(pProcData + pos1);
        int temp = pos1;
        for (int i = pos1 +1; i < pos2; i++){
            if (*(pProcData + i) <= pivot){
                temp = temp + 1;
                swap(pProcData, temp, i);
            }
        }
        swap(pProcData, pos1, temp);
        sequentialQuickSort(pProcData, pos1, temp);
        sequentialQuickSort(pProcData, temp+1, pos2);
    }
}


void ProcessTermination(double* pProcData, int ProcDataSize) {
   PrintData(pProcData, ProcDataSize);    
}