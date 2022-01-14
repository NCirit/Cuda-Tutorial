#include "common.h"
#include <stdio.h>

/**
 * @brief Compares given arrays
 * @param array1 
 * @param array2 
 * @param size Count of element in the array
*/
void compareArrays(int* array1, int* array2, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (array1[i] != array2[i])
		{
			printf("Arrays are different \n");
			return;
		}
	}
	printf("Arrays are same \n");
}