#ifdef NVCC

__global__ void sumArraysOnDevice(const float *const A, const float *const B, float *const C, const int N) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
		C[i] = A[i] + B[i];
}

__global__ void readArraysOnDevice(const float *const A, const float *const B, float *const C, const int N) {
	int i = threadIdx.x + blockIdx.x * blockDim.x, j = 0;
	if (i < N) {
		j = A[i];
		j = B[i];
	}
}
#endif /* NVCC */



#if !defined NVCC
#include <iostream>
#include <cmath>
#include <iterator>
#include <algorithm>
#include "cuda_context.h"
#include "tictoc.h"
using namespace std;

void sumArraysOnHost(const float *const A, const float *const B, float *const C, const int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
	}
}

void initialData(float *const in, const int size)
{
	double seed = toc<0>() << 20;

	srand(static_cast<unsigned int>(seed));

	for (int i = 0; i < size; i++)
	{
		in[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void checkResult(float *const ref, float *const test, const int size)
{
	double eps = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < size; i++)
	{
		if (fabs(ref[i] - test[i]) > eps)
		{
			match = 0;
			cout << "Arrays do not match!" << endl;
			cout << "ref: " << ref[i] << " test: " << test[i] << " at current " << i << endl;
			break;
		}
	}

	if (match) cout << "Arrays match.\n" << endl;
}

void printArray(float *const array, const int size) {
	copy(cbegin(array), cbegin() + size, ostream_iterator<float>(cout, "\n"));
}

void printMax(float *const array, const int size) {
	int max = 0;
	for_each(cbegin(array), cbegin(array) + size, [&max, i=0] (float n) mutable
	{
		if (n > array[max])
			max = i;
		i++;
	})
	cout << "Max " << array[max] << " at index: " << max << endl;
}

int main(int argc, char **argv) {
	startup_msg(argv[0]);

	CudaContext<0> cuda;
	cuda.module_load("cubin/sum_arrays_timed.cubin")
		.get_function("sumArraysOnDevice");

	// problem dimensions
	int N = 1 << 24;
	int nBytes = N * sizeof(float);
	dim3 block(1024);
	dim3 grid((N + block.x - 1) / block.x);
	cout << "Execution configuration <<<" << grid.x << ", " << block.x << ">>>" << endl;

	// host allocation
	float *h_A, *h_B, *h_C, *reference;
	h_A = new float[nBytes];
	h_B = new float[nBytes];
	h_C = new float[nBytes];
	reference = new float[nBytes];

	if (!(h_A && h_B && h_C && reference))
	{
		cerr << "Allocation of host arrays failed." << endl;
		goto MallocError;
	}

	// host data initialization
	initialData(h_A, N);
	initialData(h_B, N);

	// device allocation
	float *d_A, *d_B, *d_C;
	cuda_err( cudaMalloc((float **) &d_A, nBytes) );
	cuda_err( cudaMalloc((float **) &d_B, nBytes) );
	cuda_err( cudaMalloc((float **) &d_C, nBytes) );

	// copy data to device
	checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	// begin timing
	/* LARGE_INTEGER start, end, elapsedMicroseconds, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start); */


	// run kernel
	sumArraysOnDevice <<< grid, block >>>  (d_A, d_B, d_C, N);
	// readArraysOnDevice <<< grid, block >>>  (d_A, d_B, d_C, N); // checking difference with sumArraysOnDevice() to measure bandwith limits

	// check kernel errors
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	// stop timing
	// QueryPerformanceCounter(&end);
	// elapsedMicroseconds.QuadPart = ((end.QuadPart - start.QuadPart) * 1000000) / freq.QuadPart;
	// printf("On GPU, Elapsed time: %5.3f ms\n", elapsedMicroseconds.QuadPart / 1000.0f);

	// copy data to host
	checkCudaErrors(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));

	// verify calculation
	// QueryPerformanceCounter(&start);
	sumArraysOnHost(h_A, h_B, reference, N);
	// QueryPerformanceCounter(&end);
	// elapsedMicroseconds.QuadPart = ((end.QuadPart - start.QuadPart) * 1000000) / freq.QuadPart;
	// printf("On Host, Elapsed time: %5.3f ms\n", elapsedMicroseconds.QuadPart / 1000.0f);

	//checkResult(reference, h_C, N);

	//printArray(h_C, N);
	//printMax(h_C, N);

	// free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

MallocError:
	// free host memory
	free(h_A);
	free(h_B);
	free(h_C);
}