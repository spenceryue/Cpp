#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#include <iostream>

/* 	1. Reduce by min and max to get min, max, range. 
	2. Get the normalized histogram.
	3. Get the scan of the histogram.
*/

template <class I, class O>
__global__ void reduce(const I *input, const O *output, const unsigned int N)
{
	static unsigned int warp_counter = 0;
	__shared__ unsigned int warp_assignment;


	if (threadIdx.x == 0)
		atomicInc(&dartboard, );
}