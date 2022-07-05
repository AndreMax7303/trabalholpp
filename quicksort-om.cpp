#include <omp.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;

int NUM_THREADS = 8;


void dataInit(int **data, int *dataSize);
void printData(int *pData, int dataSize, int threadRank);
int **parallelHyperQuickSort(int *&data, int dataSize, int **newArrayDataSize, int **dataBase);
void LocalDataSort(int *pProcData, int ProcDataSize);
void sequentialQuickSort(int **pProcData, int pos1, int pos2);
void swap(int *pProcData, int pos1, int pos2);
int GetProcDataDivisionPos(int *pProcData, int ProcDataSize, int Pivot);
int getDatasetSize();
void PivotDistribution(int **pProcData, int *ProcDataSize, int Dim,
                       int Mask, int Iter, int *pPivot);


bool arraySortedOrNot(int *arr, int n)
{
    for (int i = 1; i < n; i++)
        if (arr[i - 1] > arr[i])
            return false;
    return true;
}

string filename = "dataset1000000.txt";
int main(int argc, char *argv[]) {
    omp_set_num_threads(NUM_THREADS);
    int dataSize;
    

    double start = omp_get_wtime();
    dataSize = getDatasetSize();
    cout << "Tamanho do dataset: " << dataSize << endl;
    int *data = new int[dataSize];
    dataInit(&data, &dataSize);
    int *newDataSizeArray;
    int *sortedData = new int[dataSize];
    parallelHyperQuickSort(data, dataSize, &newDataSizeArray, &sortedData);
    // printData(sortedData, dataSize);
    double stop = omp_get_wtime();
    cout << "Duração: " << stop - start << endl;
    if (arraySortedOrNot(sortedData, dataSize))
        cout << "Yes\n";
    else
        cout << "No\n"; 
}

void  PivotDistribution(int **pProcData, int *localDataSizeArray, int Dim,
                       int Mask, int Iter, int *pPivot)
{
    int j = 0;
    int ThreadRank = omp_get_thread_num();
    int GroupNum = NUM_THREADS / (int)pow(2, Dim - Iter);
    int *ThreadRanks = new int[GroupNum];
    int StartProc = ThreadRank - GroupNum;
    if (StartProc < 0)
    {
        StartProc = 0;
    }
    int EndProc = ThreadRank + GroupNum;
    if (EndProc > NUM_THREADS)
    {
        EndProc = NUM_THREADS;
    }
    for (int proc = StartProc; proc < EndProc; proc++)
    {
        if ((ThreadRank & Mask) >> (Iter) == (proc & Mask) >> (Iter))
        {
            ThreadRanks[j++] = proc;
        }
    }
    int firstThreadOnSubGroup = ThreadRanks[0];
    *pPivot = pProcData[firstThreadOnSubGroup][localDataSizeArray[firstThreadOnSubGroup]/2];
}


void swap(int *pProcData, int pos1, int pos2){
    int temp = *(pProcData + pos1);
    *(pProcData + pos1) = *(pProcData + pos2);
    *(pProcData + pos2) = temp;
}


void sequentialQuickSort(int **pProcData, int pos1, int pos2){
    if (pos1 < pos2){
        int pivot = *(*pProcData + pos1);
        int temp = pos1;
        for (int i = pos1 +1; i < pos2; i++){
            if (*(*pProcData + i) <= pivot){
                temp = temp + 1;
                swap(*pProcData, temp, i);
            }
        }
        swap(*pProcData, pos1, temp);
        sequentialQuickSort(pProcData, pos1, temp);
        sequentialQuickSort(pProcData, temp+1, pos2);
    }
}

void LocalDataSort(int *pProcData,int ProcDataSize){
    sequentialQuickSort(&pProcData, 0, ProcDataSize);      
}

int GetProcDataDivisionPos(int *pProcData, int ProcDataSize, int Pivot){
    int result = -1;
    for (int i = 0; i < ProcDataSize; i++) {
        if (*(pProcData + i) <= Pivot) {
            result = i;
        }
    }
    return result;
} 

void DataMerge(int *pMergeData, int MergeDataSize, int *pData, int DataSize, int *pRecvData, int RecvDataSize){
    int i = 0, j = 0, k = 0;
 
    while (i<DataSize && j <RecvDataSize) {
        if (*(pData+i) < *(pRecvData+j)){
            *(pMergeData + k) = *(pData + i);
            k+=1;
            i+=1;
        }
        else {
            *(pMergeData + k) = *(pRecvData + j);
            k +=1;
            j +=1;
        }
    }
    while (i < DataSize){
        *(pMergeData + k) = *(pData + i);
        k+=1;
        i+=1;
    }
    while (j < RecvDataSize){
        *(pMergeData + k) = *(pRecvData + j);
        k +=1;
        j +=1;
    }

}


int **parallelHyperQuickSort(int *&data, int dataSize, int **newArrayDataSize, int **dataBase) {
    int pivot;
    int HypercubeDim = (int)ceil(log(NUM_THREADS) / log(2));
    int localDataSize;
    int sendSizeArray[NUM_THREADS] = {}; 
    int restSizeArray[NUM_THREADS] = {}; 
    int localDataSizeArray[NUM_THREADS] = {}; 
    int **sendArray = new int*[NUM_THREADS];
    int **restArray = new int*[NUM_THREADS];
    
    int **dataArray = new int*[NUM_THREADS];
    for (int i=0; i < NUM_THREADS; i++){
        dataArray[i] = new int[dataSize/NUM_THREADS];
        for (int j=0; j < dataSize/NUM_THREADS; j++){
            dataArray[i][j] = *data++;
        }
        localDataSizeArray[i] = dataSize/NUM_THREADS;
    }
    #pragma omp parallel private(localDataSize, pivot)
    {
        int Mask = NUM_THREADS;
        int THREAD_RANK = omp_get_thread_num();
        localDataSize = dataSize/NUM_THREADS;
        LocalDataSort(dataArray[THREAD_RANK], dataSize / NUM_THREADS);
        for (int i = HypercubeDim; i > 0; i--) {
            #pragma omp barrier
            PivotDistribution(dataArray, localDataSizeArray, HypercubeDim, Mask, i, &pivot);
            Mask = Mask >> 1;
            int pos = GetProcDataDivisionPos(dataArray[THREAD_RANK], localDataSize, pivot);
            int threadPair, MergeDataSize;
            int *pMergeData;
            if (((THREAD_RANK&Mask) >> (i - 1)) == 0)
            {
                sendArray[THREAD_RANK] = &dataArray[THREAD_RANK][pos + 1];
                sendSizeArray[THREAD_RANK] = localDataSize - pos - 1;
                if (sendSizeArray[THREAD_RANK] < 0)
                {
                    sendSizeArray[THREAD_RANK] = 0;
                }
                threadPair = THREAD_RANK + Mask;
                restArray[THREAD_RANK] = &dataArray[THREAD_RANK][0];
                restSizeArray[THREAD_RANK] = pos + 1;
            }
            else
            { 
                sendArray[THREAD_RANK] = &dataArray[THREAD_RANK][0];
                sendSizeArray[THREAD_RANK] = pos + 1;
                if (sendSizeArray[THREAD_RANK] > localDataSize)
                {
                    sendSizeArray[THREAD_RANK] = pos;
                }
                threadPair = THREAD_RANK - Mask;
                restArray[THREAD_RANK] = &dataArray[THREAD_RANK][pos + 1];
                restSizeArray[THREAD_RANK] = localDataSize - pos - 1;
                if (restSizeArray[THREAD_RANK] < 0)
                {
                    restSizeArray[THREAD_RANK] = 0;
                }
            }
            #pragma omp barrier
            MergeDataSize = restSizeArray[THREAD_RANK] + sendSizeArray[threadPair];
            
            pMergeData = new int[MergeDataSize];
            DataMerge(pMergeData, MergeDataSize, restArray[THREAD_RANK], restSizeArray[THREAD_RANK],
                    sendArray[threadPair], sendSizeArray[threadPair]);
            dataArray[THREAD_RANK] = pMergeData;
            localDataSize = MergeDataSize;
            localDataSizeArray[THREAD_RANK] = localDataSize;   
        }   
    }
    int k =0;
    for (int i=0; i<NUM_THREADS; i++){
        for (int j=0; j<localDataSizeArray[i]; j++){
            (*dataBase)[k] = dataArray[i][j];
            k++;
        }
    }
    *newArrayDataSize = localDataSizeArray;
}

void printData(int *pData, int dataSize, int threadRank){
    string temp = "Data "+ to_string(threadRank) + ":";
    for (int i = 0; i < dataSize; i++){
        temp += to_string(*(pData+i)) + " ";
    }
    cout << temp << endl;
}

int getDatasetSize()
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

void setDataset(int *pDataset)
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

void dataInit(int **data, int *dataSize) {
    setDataset(*data);
}

