#include <iostream>
#include <cstdio>

using namespace std;


/*__host__ __device__*/ void goodbyeFromGPU() {
	// printf("Goodbye World from GPU thread %d!\n", threadIdx.x);
	printf("Goodbye World from GPU thread %d!\n", 3);
}

extern "C" {

int __declspec(dllexport) test2(const unsigned int n) {
	for (unsigned int i=0; i<n; i++) {
		cout << "Goodbye World from CPU!\n";
		
		// goodbyeFromGPU <<<1, n >>> ();
		goodbyeFromGPU();
	}
	// cudaDeviceReset();
	// cudaDeviceSynchronize();

	// Without cudaDeviceReset() or cudaDeviceSynchronize() the kernel messages are not printed.
	
	// In addition, the .exe file handle is still held by malwarebytes... sometimes.
	// Maybe only after Malwarebytes has been running a long time.
	// Restarting Malwarebytes fixes things.
	return 24;
}

}