
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
    Device code: GPU tarafindan calistirilan kod

*/
__global__ void helloWorldKernel()
{
    printf("Hello world!\n");
}

/*
    Host code: CPU tarafindan calistirilan kodumuz(main fonksiyon)
*/
int main()
{
    /*
        Grid: Bir kernel için çalıştırılan tüm thread koleksiyonudur.
        Block: Grid'de bulunan thread'ler bloklar halinde organize edilir.
        Buna thread block'ları denir. Bloklar senkronizasyon için kullanılır.

        Bir block maksimum 1024 thread'den oluşur.
        Eğer block'u belirleyen veri tipi dim3 ise;
            x <= 1024, y <= 1024 ve z <= 64
        olmalidir.

        Grid için maximum block sayisi ise;
            x <= 2^32 - 1
            y <= 65536
            z <= 65536
        şeklinde olmalıdır.

        Bu durumların aksi halinde Kernel Launch Failure ile karşılaşılır.
    
    */
    /*
        Device kodun calistirilmasi burada gerceklestirilir.
        Bu çağrı asenkrondur.
        Bu asenkron çağrı ile device kod GPU'da çalıştırılmaya başlanır.

        Bir kernel çalıştırılırken 4 adet kernel başlatma parametreleri tanımlayabiliriz.
            Kernel_name <<< number_of_blocks, thread_per_block >>> (arguments)
        Örnekte şuanda sadece 2 adet parametreyi gösteriyoruz(number_of_blocks ve thread_per_block).

        Burada number_of_blocks ve thread_per_block parametreleri int tipinde olabileceği gibi
        dim3 tipinde de olabilir.

    */
    helloWorldKernel <<< 1, 20 >>> ();
    cudaDeviceSynchronize();
    /*
    * Ornek:
        dim3 tipinde number_of_blocks ve thread_per_block
    */
    // nx: x ekseninde thread sayisi
    // ny: y ekseninde thread sayisi
    // toplam thread sayisi 16 * 4 = 64
    int nx, ny;
    nx = 16;
    ny = 4;

    /*
        dim3: x, y ve z elemanlarina sahip veri yapisi
        default x, y ve z'ye 1 atanir.
        Asagidaki ornekte;
           block degiskeni icin x=8, y=2 ve z=1'dir
    */
    dim3 block(8, 2);
    dim3 grid(nx / block.x, ny / block.y);
    helloWorldKernel << < grid, block >> > ();

    // Host kodunun calismasi; device kodun calismasi tamamlanincaya
    // kadar durdurulur.
    cudaDeviceSynchronize();

    // Kaynaklarin temizlenmesi
    cudaDeviceReset();

    return 0;
}
