
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void printThreadIds()
{
	printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d\n",
		threadIdx.x, threadIdx.y, threadIdx.z);
}


int main()
{
	/*
		Her thread'in dim3 tipinde threadIdx adli
		bir degiskeni vardir. CUDA runtime'i her thread icin
		bu degiskene tekil bir deger atar. Bu deger thread'in
		thread block'undaki yerine gore belirlenir.

		Ornegin asagidaki gibi tek boyutlu bir thread bloğu için;
			
					| A | B | C | D | E | F | G | H |

		threadIdx.x	  0   1   2   3   4   5   6   7
		threadIdx.y	  0   0   0   0   0   0   0   0
		threadIdx.z	  0   0   0   0   0   0   0   0

		şeklinde olur. Örneğin F threadinin threadIdx'i
			threadIdx.x = 5, threadIdx.y = 0, threadIdx.z = 0	
		şeklindedir.
	
	*/

	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8, 8);
	dim3 grid(nx / block.x, ny / block.y);

	printThreadIds << < grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
	
}
