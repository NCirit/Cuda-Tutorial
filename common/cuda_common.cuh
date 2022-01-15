#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

/// Error cheking macro
#define CudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}

/**
 * @brief Error handling function for the cuda functions
 * @param code Return value of the cuda function call
 * @param file Name of the file that the function is called
 * @param line Line number of the call
 * @param abort Should exit or not
*/
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}