#include <iostream>
#include <cstdio>

using namespace std;

extern "C" __global__ void helloFromGPU() {
	printf("Hello World from GPU thread %d!\n", threadIdx.x);
}


 /*int __declspec(dllexport) test(const unsigned int n) {
	for (unsigned int i=0; i<n; i++) {
		cout << "Hello World from CPU!\n";
		
		helloFromGPU <<<1, n >>> ();
	}
	// cudaDeviceReset();
	cudaDeviceSynchronize();

	// Without cudaDeviceReset() or cudaDeviceSynchronize() the kernel messages are not printed.
	
	// In addition, the .exe file handle is still held by malwarebytes... sometimes.
	// Maybe only after Malwarebytes has been running a long time.
	// Restarting Malwarebytes fixes things.
	return 42;
}
*/