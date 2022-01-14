
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void memoryTransferExample(int *input)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid: %d, gid: %d, value: %d \n",
        threadIdx.x, gid, input[gid]);
}


class k {

};
int main()
{
    /*
           Bir CUDA programin calismasi basitce;
            1- Host tarafinda verinin initilize edilmesi
            2- Veri uzerinde yapilacak olan yogun islemin gerceklestirlmesi
               gorevinin GPU'ya verilmesi
            3- Bu esnada Host kendi calismasina devam eder. GPU'da yapilan
               islemlerin sonucu istediginde, GPU'nun gorevini bitirmesini bekler.

            Bellek Host ve Device icin fiziksel olarak ayri oldugundan dolayi, 
            yukaridaki islemde;
                1- Verinin GPU'ya aktarilmasi
                2- Sonuc verisinin Host'a aktarilmasi
            olmak uzere iki adet bellek transferi vardir.

            Asagidaki fonksiyon kullanilarak host ile device arasinda
            veri transferi gerceklestirilebilir.
                cudaMemCpy( destinationPtr, sourcePtr, sizeInBytes, direction)

                Burada direction; HostToDevice, DeviceToHost veya DeviceToDevice 
                olabilir.

            C           CUDA
            malloc      cudaMalloc
            memset      cudaMemset
            free        cudaFree
    */

    int size = 128;
    int sizeInBytes = size * sizeof(int);

    int* hostMemory;
    hostMemory = (int*)malloc(sizeInBytes);

    time_t t;
    // seeding rand function
    srand((unsigned)time(&t));
    for (size_t i = 0; i < size; i++)
    {
        hostMemory[i] = (int)(rand() & 0xFF);
    }

    int* deviceMemory;
    cudaMalloc((void**)&deviceMemory, sizeInBytes);

    cudaMemcpy(deviceMemory, hostMemory, sizeInBytes, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(2);

    memoryTransferExample << <grid, block >> > (deviceMemory);
    cudaDeviceSynchronize();

    cudaFree(deviceMemory);
    free(hostMemory);

    cudaDeviceReset();
    return 0;
}
