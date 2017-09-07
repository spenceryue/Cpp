#include <stdio.h>

__global__ void helloFromGPU() {
	printf("Hello World from GPU thread %d!\n",threadIdx.x);
}

int main() {
	printf("Hello World from CPU!\n");
	
	helloFromGPU <<<1, 10 >>> ();
	cudaDeviceReset();
	//cudaDeviceSynchronize();

	// Without cudaDeviceReset() or cudaDeviceSynchronize() the kernel messages are not printed.
	
	// In addition, the .exe file handle is still held by malwarebytes... sometimes.
	// Maybe only after Malwarebytes has been running a long time.
	// Restarting Malwarebytes fixes things.
	return 0;
}