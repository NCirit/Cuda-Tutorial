
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void assignment()
{
    printf("threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d, "
        "blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d ,"
        "gridDim.x = %d, gridDim.y = %d, gridDim.z = %d\n"
        ,
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        gridDim.x, gridDim.y, gridDim.z);
}

int main()
{

    /*
    *   Her eksende 4 thread'e sahip bir grid olustur.
    *   Block size her eksende 2 thread olacak sekilde olmalidir.
        Bu degerleri ekrana bastir: threadIdx, blockIdx, gridDim
    
    */
    
    int nx, ny, nz;
    nx = 4;
    ny = 4;
    nz = 4;

    dim3 block(2, 2, 2);
    dim3 grid(nx / block.x, ny / block.y, nz / block.z);

    assignment << < grid, block >> > ();


    cudaError_t cudaStatus;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
