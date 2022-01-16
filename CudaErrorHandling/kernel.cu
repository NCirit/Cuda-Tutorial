
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"
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
__global__ void sumArraysKernel(int* a, int* b, int* c, int size)
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

    /*
        Genel olarak endüstriyal cuda uygulamalarda;
            Execution time
            Power consumption
            Floor space
            Cost of hardware
        gibi kriterler performasn ölçümünde kullanılır.
    
    */

    /*
        Trail and Error yöntemi

        Bu yöntemde aynı kernel farklı blok konfigrasyonları
        ile çalıştırılarak, execution time'lar hesaplanır.
        Bu sayede farklı konfigrasyonlardan execution time'ı en
        düşük konfigrasyon seçilir.
    
    */

    int size = 100000;

    int blockSize = 128;

    int numberOfBytes = size * sizeof(int);

    int* pHostA, * pHostB, * pHostResultsFromGPU, * pHostResultsFromCPU;

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

    CudaErrorCheck(cudaMalloc((int**)&pDeviceA, numberOfBytes));
    CudaErrorCheck(cudaMalloc((int**)&pDeviceB, numberOfBytes));
    CudaErrorCheck(cudaMalloc((int**)&pDeviceResult, numberOfBytes));

    // Ilgili islemin ne kadar surdugunu ogrenmek icin
    clock_t cpuStart, cpuEnd;
    cpuStart = clock();
    sumArraysOnCpu(pHostA, pHostB, pHostResultsFromCPU, size);
    cpuEnd = clock();

    clock_t hostToDeviceStart, hostToDeviceEnd;
    hostToDeviceStart = clock();
    CudaErrorCheck(cudaMemcpy(pDeviceA, pHostA, numberOfBytes, cudaMemcpyHostToDevice));
    CudaErrorCheck(cudaMemcpy(pDeviceB, pHostB, numberOfBytes, cudaMemcpyHostToDevice));
    hostToDeviceEnd = clock();

    dim3 block(blockSize);

    // Burada size'in block.x'e tam bolunmemesi
    // durumu icin fazladan 1 ekliyoruz. Bu sayede
    // grid icinde fazladan 1 blok daha oluyor.
    // Bundan dolayi calistirilan kod icinde
    // gid hesaplanirken ilgili dizinin sinirlari
    // dahilinde olunup olunmadiginin kontrol edilmesi
    // gerekiyor.
    dim3 grid((size / block.x) + 1);

    clock_t gpuStart, gpuEnd;
    gpuStart = clock();
    sumArraysKernel << <grid, block >> > (pDeviceA, pDeviceB, pDeviceResult, size);
    // Kernelin calismasinin bitmesini bekle
    CudaErrorCheck(cudaDeviceSynchronize());
    gpuEnd = clock();

    clock_t deviceToHostStart, deviceToHostEnd;
    deviceToHostStart = clock();
    // GPU'dan sonuclari geri CPU memory'sine aktar
    CudaErrorCheck(cudaMemcpy(pHostResultsFromGPU, pDeviceResult, numberOfBytes, cudaMemcpyDeviceToHost));
    deviceToHostEnd = clock();


    compareArrays(pHostResultsFromCPU, pHostResultsFromGPU, size);


    printf("Sum array CPU execution time: %4.6f \n",
        (double)((double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC));

    printf("Sum array GPU execution time: %4.6f \n",
        (double)((double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC));

    printf("Host to device memory transfer time: %4.6f \n",
        (double)((double)(hostToDeviceEnd - hostToDeviceStart) / CLOCKS_PER_SEC));

    printf("Device to host memory transfer time: %4.6f \n",
        (double)((double)(deviceToHostEnd - deviceToHostStart) / CLOCKS_PER_SEC));

    printf("GPU total execution time: %4.6f \n",
        (double)((double)(deviceToHostEnd - hostToDeviceStart) / CLOCKS_PER_SEC));

    cudaFree(pDeviceA);
    cudaFree(pDeviceB);
    cudaFree(pDeviceResult);


    free(pHostA);
    free(pHostB);
    free(pHostResultsFromGPU);
    free(pHostResultsFromCPU);



    CudaErrorCheck(cudaDeviceReset());
    return;

}
