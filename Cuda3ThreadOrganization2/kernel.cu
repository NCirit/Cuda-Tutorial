
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printDetails()
{
    printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d,\
             blockDim.x: %d, blockDim.y: %d,blockDim.z: %d,\
             gridDim.x: %d, gridDim.y: %d, gridDim.z: %d,\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}


int main()
{
    /*
        Her thread'in blockIdx adli dim3 tipinde
        bir degiskeni vardir. Bu degiskene, CUDA runtime
        tarafindan thread'in ait oldugu blogun id'si(aslinda
        ilgili blogun grid icindeki pozisyonu) atanir.


        Ornegin, asagidaki gibi iki adet thread blogumuz oldugunu dusunuelim
            
                        | A | B | C | D |             | E | F | G | H |

        blockIdx.x        0   0   0   0                 1   1   1   1 
        blockIdx.y        0   0   0   0                 0   0   0   0
        blockIdx.z        0   0   0   0                 0   0   0   0
    
        seklinde olur.

        Her thread'in ayrica blockDim adli dim3 tipinde
        bir degiskeni vardir. Bu degisken bir bloga ait 
        thread sayisini tutar(x, y ve z icin ayri ayri).

        Ornegin yukaridaki ornek icin;
            blockDim.x = 4,
            blockDim.y = 1'dir.

        Not: Bir grid'deki tum thread blocklari ayni boyutlardadir.
        Yani grid icindeki tum thread'ler icin bu deger aynidir.

        GridDim degiskeni dim3 tipindedir. Bu degisken
        grid'e ait her eksendeki thread blogu sayisini belirler.
    
    */

    /*
        Ornek: Toplam 256 adet thread'den olusan bir grid;
        GridDim(2,2,1) olacak sekilde 4 adet blocktan olussun.
        Her bir block (8,8,1) sekilde tanimlansin.
    */

    int nx, ny;
    nx = 16;
    ny = 16;

    dim3 block(8, 8);
    dim3 grid(nx / block.x, ny / block.y);

    printDetails << < grid, block >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
