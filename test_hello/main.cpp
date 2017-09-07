#include <iostream>
// #include "../tictoc.h"
using namespace std;

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
	__declspec(dllexport) __global__ void helloFromGPU();
	// __declspec(dllimport) int test(unsigned int);
	// __declspec(dllimport) int test2(unsigned int);
}

int main(int argc, char** argv)
{
	cout << argv[0] << "...Starting\n" << endl;

	/*cout << "Calling test...\n\n";
	cout << tictoc<test>(3) << endl;
	cout << "Done." << endl;*/

	for (unsigned int i=0; i<n; i++) {
		cout << "Hello World from CPU!\n";
		
		helloFromGPU <<<1, n >>> ();
	}
	// cudaDeviceReset();
	cudaDeviceSynchronize();

	// cout << "Calling test2...\n\n";
	// cout << tictoc<test2>(3) << endl;
	// cout << "Done." << endl;
}