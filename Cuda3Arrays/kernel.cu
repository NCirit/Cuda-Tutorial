
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void uniqueIndex(int* input)
{
    int tid = threadIdx.x;
    printf("threadIdx: %d, value: %d \n", tid, input[tid]);
}

__global__ void uniqueIndex2(int* input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;
    printf("threadIdx: %d, value: %d \n", tid, input[gid]);
}

int main()
{
    /*
        Bir CUDA programinda array indexleri olarak
        threadIdx, blockIdx, blockDim, gridDim yaygin olarak kullanilir.
    
    */

    /*
        Ornek: 8 Elemanli bir dizimiz olsun. Ayrica 8 thread bulunan bir
        gridimiz olsun. Bu dizinin her bir elemanina sadece tek bir thread 
        erisebilecek sekilde kodlama yapalim.
    */
    int arraySize = 8;
    int arrayByteSize = sizeof(int) * arraySize;
    int hData[] = { 23,9,4,53,65,12,1,33 };

    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", hData[i]);
    }
    printf("\n \n");
   
    int* dData;

    cudaMalloc((void**)&dData, arrayByteSize);
    cudaMemcpy(dData, hData, arrayByteSize, cudaMemcpyHostToDevice);

    dim3 block(8);
    dim3 grid(1);

    uniqueIndex << < grid, block >> > (dData);
    cudaDeviceSynchronize();

    /*
        Yukaridaki ornegimiz tek boyutlu bir gridden olusuyordu.
        Yani sadece tek bir gridimiz vardi.
        Ornegin asagidaki gibi bir block yapimiz oldugunu varsayalim.

                        | A | B | C | D |       | E | F | G | H |

        threadIdx.x       0   1   2   3           0   1   2   3
        offset          0                       4
        şeklinde 2 thread blogundan olusan bır grid yapimiz olsun.
        Burada threadIdx thread'in ilgili bloktaki pozisyonu oldugu icin
        A ile E'nin, B ile F'nin ... seklinde threadIdx degerleri ayni olur.
        Bundan dolayi dizide ekrana bastirdigimiz fonksiyonda iki ayni threadIdx'lere
        sahip thread'ler ayni elemani ekrana bastiracaktir.

        Bunu onlemek icin indeks hesaplarken bloga gore offset ekleyebiliriz:
            gid = tid + offset
            gid = tid + blockIdx.x * blockDim.x

        Buradaki offset degeri, blogun grid uzerindeki pozisyonu ve boyutu
        ile hesaplanir. Bu sayede istedigimiz sonucu elde ederiz.
    */
    dim3 block1(4);
    dim3 grid1(2);
    printf("\n \n");
    uniqueIndex << < grid, block >> > (dData);
    cudaDeviceSynchronize();

    /*
        Yukaridaki ornekteki offset degerine block offset'i olarak adlandirabiliriz.

        Ornegin gridimiz tek boyutulu olmak yerine 2 boyutlu oldugunu dusunelim.
        Yani grid'in block sayisinin x icin n, y icin m tane oldugunu.
        Boyle oldugunda daha onceden oldugu gibi bazi thread'ler ayni
        dizi elemanlarina erisecektir. Bundan dolayi indeks hesaplamamiza
        yeni bir terim daha eklemememiz gereklidir. Buna row offset(griddeki her
        satirda bulunan thread sayisi) diyelim. Yeni indeks hesaplamamiz;
            gid = rowOffset + blockOffset + threadidx.x
            blockOffset = blockIdx.X * blockDim.x
            rowOffset = gridDim.x * blockDim.x * blockIdx.y
        seklinde  olur.
    
    */

    /*
        Genel olarak biz ayni blockta olan thread'lerin
        ardisik olarak dizi elemanlarina erismesini isteriz.
        Ancak yukaridaki ornekte bu saglanmadi.
    
    */

    cudaDeviceReset();


    return 0;
}

