#include <mpi.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

// https://drive.google.com/file/d/1kabOOoDFdZ8lEUhgJ1VcT5o8xP1pA3pG/view
// https://iq.opengenus.org/parallel-quicksort/

void PivotDistribution(int *pProcData, int ProcDataSize, int Dim,
                       int Mask, int Iter, int *pPivot);
int *ParallelHyperQuickSort(int *pProcData, int *ProcDataSize);
int getDatasetSize(string filename);
void setDataset(string filename, int *pDataset);
void ProcessInitialization(int **pProcData, int *ProcDataSize);
void ProcessTermination(int *pProcData, int ProcDataSize);
void LocalDataSort(int *pProcData, int ProcDataSize);
void swap(int *pProcData, int pos1, int pos2);
void sequentialQuickSort(int *pProcData, int pos1, int pos2);
int GetProcDataDivisionPos(int *pProcData, int ProcDataSize, int Pivot);

// The HyperQuickSort Method
int ProcRank; // Rank of current process
int ProcNum;  // Number of processes
double start;
double stop;

bool arraySortedOrNot(int *arr, int n)
{
    // Array has one or no element
    if (n == 0 || n == 1)
        return true;
 
    for (int i = 1; i < n; i++)
 
        // Unsorted pair found
        if (arr[i - 1] > arr[i])
            return false;
 
    // No unsorted pair found
    return true;
}

int main(int argc, char *argv[])
{
    int *pProcData;   // Data block for the process
    int ProcDataSize; // Data block size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    if (ProcRank == 0)
    {
        start = MPI_Wtime();
    }
    // Data Initialization and their distribution among the processors
    ProcessInitialization(&pProcData, &ProcDataSize);

    // Parallel sorting
    pProcData = ParallelHyperQuickSort(pProcData, &ProcDataSize);
    // The termination of process computations
    ProcessTermination(pProcData, ProcDataSize);
    if (ProcRank == 0)
    {
        stop = MPI_Wtime();
        cout << "Duração: " << stop - start << endl;
    }
    MPI_Finalize();
}

// Determination of the pivot value and broadcast it to all the processors
void PivotDistribution(int *pProcData, int ProcDataSize, int Dim,
                       int Mask, int Iter, int *pPivot)
{
    MPI_Group WorldGroup;
    MPI_Group SubcubeGroup; // a group of processors – a subhypercube
    MPI_Comm SubcubeComm;   // subhypercube communcator
    int j = 0;
    int GroupNum = ProcNum / (int)pow(2, Dim - Iter);
    int *ProcRanks = new int[GroupNum];
    // Forming the list of ranks for the hypercube processes
    int StartProc = ProcRank - GroupNum;
    if (StartProc < 0)
    {
        StartProc = 0;
    }
    int EndProc = ProcRank + GroupNum;
    if (EndProc > ProcNum)
    {
        EndProc = ProcNum;
    }
    // if (Iter == 1)
    // {
    //     cout << "Proc " << ProcRank << "- GroupNum: " << GroupNum << " StartProc: " << StartProc << " EndProc: " << EndProc << endl;
    // }
    for (int proc = StartProc; proc < EndProc; proc++)
    {
        if ((ProcRank & Mask) >> (Iter) == (proc & Mask) >> (Iter))
        {
            ProcRanks[j++] = proc;
        }
    }
    // Creating the communicator for the subhypercube processes
    MPI_Comm_group(MPI_COMM_WORLD, &WorldGroup);
    MPI_Group_incl(WorldGroup, GroupNum, ProcRanks, &SubcubeGroup);
    MPI_Comm_create(MPI_COMM_WORLD, SubcubeGroup, &SubcubeComm);
    // Selecting the pivot element and seding it to the subhypercube processes
    if (ProcRank == ProcRanks[0])
    {
        *pPivot = pProcData[(ProcDataSize) / 2];
        // cout << "Pivot: " << *pPivot << endl;
    }
    MPI_Bcast(pPivot, 1, MPI_INT, 0, SubcubeComm);
    MPI_Group_free(&SubcubeGroup);
    MPI_Comm_free(&SubcubeComm);
    delete[] ProcRanks;
}

int GetProcDataDivisionPos(int *pProcData, int ProcDataSize, int Pivot)
{
    int result = -1;
    for (int i = 0; i < ProcDataSize; i++)
    {
        if (*(pProcData + i) <= Pivot)
        {
            result = i;
        }
    }
    return result;
}

void DataMerge(int *pMergeData, int MergeDataSize, int *pData, int DataSize, int *pRecvData, int RecvDataSize)
{
    int i = 0, j = 0, k = 0;

    while (i < DataSize && j < RecvDataSize)
    {
        if (*(pData + i) < *(pRecvData + j))
        {
            *(pMergeData + k) = *(pData + i);
            k += 1;
            i += 1;
        }
        else
        {
            *(pMergeData + k) = *(pRecvData + j);
            k += 1;
            j += 1;
        }
    }
    while (i < DataSize)
    {
        *(pMergeData + k) = *(pData + i);
        k += 1;
        i += 1;
    }
    while (j < RecvDataSize)
    {
        *(pMergeData + k) = *(pRecvData + j);
        k += 1;
        j += 1;
    }
}

void PrintData(int *pData, int dataSize)
{
    string temp = "Proc " + to_string(ProcRank) + ":";
    for (int i = 0; i < dataSize; i++)
    {
        temp += to_string(*(pData + i)) + " ";
    }
    cout << temp << endl;
}

// // The Parallel HyperQuickSort Method
int *ParallelHyperQuickSort(int *pProcData, int *ProcDataSize)
{
    MPI_Status status;
    int CommProcRank; // Rank of the processor involved in communications
    int *pMergeData,  // Block obtained after merging the block parts
        *pData,       // Block part, which remains on the processor
        *pSendData,   // Block part, which is sent to the processor CommProcRank
        *pRecvData;   // Block part, which is received from the proc CommProcRank
    int DataSize, SendDataSize, RecvDataSize, MergeDataSize;
    int HypercubeDim = (int)ceil(log(ProcNum) / log(2)); // Hypercube dimension
    int Mask = ProcNum;
    int Pivot;
    // Local data sorting
    LocalDataSort(pProcData, *ProcDataSize);
    // Hyperquicksort iterations
    for (int i = HypercubeDim; i > 0; i--)
    {
        // Determination of the pivot value and broadcast it to processors
        PivotDistribution(pProcData, *ProcDataSize, HypercubeDim, Mask, i, &Pivot);
        Mask = Mask >> 1;
        // Determination of the data division position
        int pos = GetProcDataDivisionPos(pProcData, *ProcDataSize, Pivot);
        // Block division
        if (((ProcRank & Mask) >> (i - 1)) == 0)
        { // high order bit= 0
            pSendData = &pProcData[pos + 1];
            SendDataSize = *ProcDataSize - pos - 1;
            if (SendDataSize < 0)
            {
                SendDataSize = 0;
            }
            CommProcRank = ProcRank + Mask;
            pData = &pProcData[0];
            DataSize = pos + 1;
        }
        else
        { // high order bit = 1
            pSendData = &pProcData[0];
            SendDataSize = pos + 1;
            if (SendDataSize > *ProcDataSize)
            {
                SendDataSize = pos;
            }
            CommProcRank = ProcRank - Mask;
            pData = &pProcData[pos + 1];
            DataSize = *ProcDataSize - pos - 1;
            if (DataSize < 0)
            {
                DataSize = 0;
            }
        }
        // Sending the sizes of the data block parts
        MPI_Sendrecv(&SendDataSize, 1, MPI_INT, CommProcRank, 0,
                     &RecvDataSize, 1, MPI_INT, CommProcRank, 0, MPI_COMM_WORLD, &status);
        // Sending the data block parts
        pRecvData = new int[RecvDataSize];
        MPI_Sendrecv(pSendData, SendDataSize, MPI_INT,
                     CommProcRank, 0, pRecvData, RecvDataSize, MPI_INT,
                     CommProcRank, 0, MPI_COMM_WORLD, &status);
        // Data merge
        MergeDataSize = DataSize + RecvDataSize;
        pMergeData = new int[MergeDataSize];
        DataMerge(pMergeData, MergeDataSize, pData, DataSize,
                  pRecvData, RecvDataSize);
        // delete[] pProcData;
        // delete[] pRecvData;
        pProcData = pMergeData;
        *ProcDataSize = MergeDataSize;
        // PrintData(pProcData, *ProcDataSize);
    }
    return pProcData;
}

int getDatasetSize(string filename)
{
    ifstream myFile;
    string temp;

    myFile.open(filename);

    int count = 0;
    while (myFile >> temp)
    {
        count++;
    }
    myFile.close();
    return count;
}

void setDataset(string filename, int *pDataset)
{
    ifstream myFile;
    string temp;
    myFile.open(filename);
    int count = 0;
    while (myFile >> temp)
    {
        pDataset[count] = stod(temp);
        count++;
    }
    myFile.close();
}

void ProcessInitialization(int **pProcData, int *ProcDataSize)
{
    string filename = "dataset1000000.txt";
    int datasetSize;
    int *dataset;
    if (ProcRank == 0)
    {
        datasetSize = getDatasetSize(filename);
        cout << "Tamanho do dataset: " << datasetSize << endl;
        dataset = new int[datasetSize];
        setDataset(filename, dataset);
        *ProcDataSize = datasetSize / ProcNum;
    }
    MPI_Bcast(ProcDataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *pProcData = new int[*ProcDataSize];
    MPI_Scatter(dataset, *ProcDataSize, MPI_INT, *pProcData, *ProcDataSize, MPI_INT, 0, MPI_COMM_WORLD);
}

void swap(int *pProcData, int pos1, int pos2)
{
    int temp = *(pProcData + pos1);
    *(pProcData + pos1) = *(pProcData + pos2);
    *(pProcData + pos2) = temp;
}

void sequentialQuickSort(int *pProcData, int pos1, int pos2)
{
    if (pos1 < pos2)
    {
        int pivot = *(pProcData + pos1);
        int temp = pos1;
        for (int i = pos1 + 1; i < pos2; i++)
        {
            if (*(pProcData + i) <= pivot)
            {
                temp = temp + 1;
                swap(pProcData, temp, i);
            }
        }
        swap(pProcData, pos1, temp);
        sequentialQuickSort(pProcData, pos1, temp);
        sequentialQuickSort(pProcData, temp + 1, pos2);
    }
}

void LocalDataSort(int *pProcData, int ProcDataSize)
{
    sequentialQuickSort(pProcData, 0, ProcDataSize);
}

void ProcessTermination(int *pProcData, int ProcDataSize)
{
    int *procDataSizeArray = new int[ProcNum];
    MPI_Gather(&ProcDataSize, 1, MPI_INT, procDataSizeArray, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int totalSize = 0;
    int displacement[ProcNum] = {};
    int *sortedProcData;
    for (int i = 0; i < ProcNum; i++)
    {
        totalSize += *(procDataSizeArray + i);
    }

    if (ProcRank == 0)
    {
        sortedProcData = new int[totalSize];
        for (int i = 1; i < ProcNum; i++)
        {
            displacement[i] = displacement[i - 1] + *(procDataSizeArray + i - 1);
        }
    }
    MPI_Gatherv(pProcData, ProcDataSize, MPI_INT, sortedProcData, procDataSizeArray, displacement, MPI_INT, 0, MPI_COMM_WORLD);
    if (ProcRank ==0){
        if (arraySortedOrNot(sortedProcData, totalSize))
        cout << "Yes\n";
    else
        cout << "No\n";
    }
    
    // if (ProcRank == 0){
    //     PrintData(sortedProcData, totalSize);
    // }
}