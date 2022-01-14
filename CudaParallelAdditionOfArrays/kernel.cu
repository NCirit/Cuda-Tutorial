
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include <stdio.h>

// For random initialization
#include <stdlib.h>
#include <time.h>

// for memset
#include <cstring>


/**
 * @brief Kernel function that will be executed on GPU
 * @param c Pointer to an array that resides on GPU. Results
 * will be stored in this array.
 * @param a Pointer to an array that resides on GPU.
 * @param b Pointer to an array that resides on GPU.
 * @param size Element count of the array
 * @return 
*/
__global__ void sumArraysKernel(int *a, int *b, int *c, int size)
{
    // Our block is one dimentional
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

/**
 * @brief Performs element wise summation of given arrays and puts back
 * result to another array.
 * @param c The array which results are put back into.
 * @param a 
 * @param b 
 * @param size Element count of the arrays
*/
void sumArraysOnCpu(int* a, int* b, int* c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int size = 10000;

    int blockSize = 128;

    int numberOfBytes = size * sizeof(int);

    int* pHostA, * pHostB, * pHostResultsFromGPU, *pHostResultsFromCPU;

    pHostA = (int*)malloc(numberOfBytes);
    pHostB = (int*)malloc(numberOfBytes);
    pHostResultsFromGPU = (int*)malloc(numberOfBytes);
    pHostResultsFromCPU = (int*)malloc(numberOfBytes);

    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        pHostA[i] = (int)(rand() & 0xFF);
    }

    for (int i = 0; i < size; i++)
    {
        pHostB[i] = (int)(rand() & 0xFF);
    }


    memset(pHostResultsFromGPU, 0, numberOfBytes);



    int* pDeviceA, * pDeviceB, * pDeviceResult;

    cudaMalloc((int**)&pDeviceA, numberOfBytes);
    cudaMalloc((int**)&pDeviceB, numberOfBytes);
    cudaMalloc((int**)&pDeviceResult, numberOfBytes);

    cudaMemcpy(pDeviceA, pHostA, numberOfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceB, pHostB, numberOfBytes, cudaMemcpyHostToDevice);


    dim3 block(blockSize);

    // Burada size'in block.x'e tam bolunmemesi
    // durumu icin fazladan 1 ekliyoruz. Bu sayede
    // grid icinde fazladan 1 blok daha oluyor.
    // Bundan dolayi calistirilan kod icinde
    // gid hesaplanirken ilgili dizinin sinirlari
    // dahilinde olunup olunmadiginin kontrol edilmesi
    // gerekiyor.
    dim3 grid((size / block.x) + 1);

    sumArraysKernel << <grid, block >> > (pDeviceA, pDeviceB, pDeviceResult, size);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    // Kernelin calismasinin bitmesini bekle
    cudaDeviceSynchronize();

    // GPU'dan sonuclari geri CPU memory'sine aktar
    cudaMemcpy(pHostResultsFromGPU, pDeviceResult, numberOfBytes, cudaMemcpyDeviceToHost);

    sumArraysOnCpu(pHostA, pHostB, pHostResultsFromCPU, size);


    compareArrays(pHostResultsFromCPU, pHostResultsFromGPU, size);


    cudaFree(pDeviceA);
    cudaFree(pDeviceB);
    cudaFree(pDeviceResult);


    free(pHostA);
    free(pHostB);
    free(pHostResultsFromGPU);
    free(pHostResultsFromCPU);
    


    cudaDeviceReset();
    return;

}
