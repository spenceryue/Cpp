#ifdef NVCC
	#include <cstdio>

extern "C" __global__ void kernel() {
	printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

extern "C" __global__ void saxpy(const float a, const float *x, const float *y, float *result, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        result[i] = a*x[i] + y[i];
		// printf("%.2f * %f + %f = %f\n", a, x[i], y[i], result[i]);
	}
}

#else
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include <iostream> 		// std::cout, std::cerr, std:: endl
#include <iomanip>	 		// std::setw, std::left
#include <string> 			// std::string
#include <forward_list>		// std::forward_list
#include <algorithm>		// std::find, std::for_each
#include <type_traits>		// std::enable_if_t, std::is_pointer_v, std::remove_reference_t
#include <utility>			// std::forward, std::declval
#include <exception>		// std::exception
#include "basename.h"		// filename
#include "read_file.h"
#include "report_errors.h"	// throw_err, warn_err
#include "faces.h"			// blank_face, check_mark
#include "type_stuff.h"		// type_names
using namespace std::string_literals;

#include <cuda.h>
#undef __cdecl
#include <builtin_types.h>
/*
CUdevice (this is just a typedef for 'int' lol)
CUresult (an 'enum' of error codes)
	- CUDA_SUCCESS = 0
CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuGetErrorName(CUresult error, const char **pStr);
CUresult cuGetErrorString(CUresult error, const char **pStr);
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
	CU_CTX_SCHED_AUTO = 0
CUresult cuModuleLoad(CUmodule *module, const char *fname);
CUjit_option
CUresult cuModuleLoad(CUmodule *module, const char *fname);
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra);
CUresult cuCtxSynchronize(void);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
*/

#define cuda_err(except, expr)			do {error = checkCudaErrors<except>( (expr), #expr, __FILE__, __LINE__);} while(0)
#define cuda_warn(expr)					do {error = checkCudaErrors<false>( (expr), #expr, __FILE__, __LINE__);} while(0)
template<bool except = true>
inline bool checkCudaErrors (CUresult err, const std::string& func, const std::string& file, int line)
{
	if (err != CUDA_SUCCESS)
	{
		using namespace std;
		cout << flush;

		char *name = 0, *message = 0;
		cuGetErrorName(err, const_cast<const char**>(&name));
		cuGetErrorString(err, const_cast<const char**>(&message));
		
		*name = name ? *name : 0;
		*message = message ? *message : 0;

		cerr
			<< "Oh dear. " << blank_face << "\n"
			<< "A CUDA error [" << name << "] occurred in \"" << file << "\" at line: " << line << "\n"
			<< func << "    -->    " << message << "\n"
			<< endl;

		if constexpr(except)
			throw exception();
		
		return true;
	}

	return false;
}

template <
bool verbose = true,
bool except = true>
class CudaContext
{
public:
	int device_number = 0;
	std::string name;
	std::string kernel_name;
	
	CUdevice dev = 0;
	CUcontext context = NULL;
	CUmodule module = NULL;
	CUfunction kernel = NULL;
	
	bool error = false;
	
private:
	std::forward_list<CUdeviceptr> to_destroy;

	void init (unsigned int device_number = 0)
	{
		// cuInit() must happen first
		cuda_err(except, cuInit(0) );
		
		// get device handle
		cuda_err(except, cuDeviceGet(&dev, device_number) );
		
		// get GPU's name
		if constexpr (verbose)
		{
			name.resize(64);
			// ^ only resize if verbose is set to true
			cuda_err(except, cuDeviceGetName(&name[0], 64, device_number) );
			std::cout << std::left << std::setw(60) << std::setfill(' ')
					  << "> Using CUDA Device [" + std::to_string(device_number) + "]: " + name.c_str() << check_mark
					  // ^ c_str() necessary to avoid printing NULL character
					  << std::endl;
		}
	}

public:
	CudaContext (unsigned int device_number = 0)
	{
		init(device_number);

		cuda_err(except, cuCtxCreate(&context, CU_CTX_SCHED_AUTO, dev) );

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> CudaContext successfully created." << check_mark
						  << std::endl;
	}

	CudaContext (const std::string& module_path) :
		CudaContext ()
	{
		module_load(module_path);
	}

	CudaContext (const std::string& module_path, const std::string& kernel_name) :
		CudaContext (module_path)
	{
		get_kernel(kernel_name);
	}

	~CudaContext ()
	{
		int freed = 0,
			count = 0;

		std::for_each (begin(to_destroy), end(to_destroy), [&] (auto ptr)
		{
			if (ptr == 0) // "erased" ptrs are set to 0
				return;

			count++;
			cuda_warn( cuMemFree(ptr) );
			if (!error)
				freed++;
		});

		if (int remaining = count - freed; remaining)
			warn_err("Warning: " + std::to_string(remaining) + " device pointers were not successfully freed.");
		else if constexpr (verbose)
			std::cout << std::left << std::setw(60) << std::setfill(' ')
					  << "> All device memory allocations were successfully freed." << check_mark
					  << std::endl;

		cuda_warn( cuCtxDestroy(context) );
		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> CudaContext successfully destroyed." << check_mark
						  << std::endl;
	}

	auto& module_load(const std::string& module_path)
	{
		std::string file(filename(module_path));

		// looking for .ptx, .cubin, or .fatbin
		if ( file.rfind("ptx") != file.npos )
			jit_module_load(module_path);
		else if ( file.rfind("cubin") != file.npos ||
				  file.rfind("fatbin") != file.npos )
			bin_module_load(module_path);
		else
			throw_err("Oops! File" + file + "should have .ptx, .cubin, or .fatbin extension.");

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> Module \"" + module_path + "\" successfully loaded." << check_mark
						  << std::endl;
		
		return *this;
	}

	auto& jit_module_load (const std::string& module_path)
	{
		std::string ptx_source = read_file(module_path);
		// ^ must read file in binary mode on Windows because '\r' characters prevent compilation.

		if constexpr(verbose)
			std::cout << std::left << std::setw(60) << std::setfill(' ')
					  << "> File \"" + module_path + "\" successfully read." << check_mark
					  << std::endl;

		constexpr unsigned int num_options = 5;
		
		// specify options to parameterize
		CUjit_option options[num_options]
		{
			CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_INFO_LOG_BUFFER,
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_ERROR_LOG_BUFFER,
			CU_JIT_MAX_REGISTERS
		};
		
		// give option values
		unsigned int info_buffer_size = 1024;
		std::string info_buffer(info_buffer_size, 0);
		unsigned int err_buffer_size = 1024;
		std::string err_buffer(err_buffer_size, 0);
		unsigned int max_registers = 32;

		// prepare pointers to values
		void* values[num_options]
		// ^ void* is a dumby type to allow passing various option value types. It is not a pointer type.
		{
			reinterpret_cast<void*>(info_buffer_size),
			reinterpret_cast<void*>(&info_buffer[0]),
			reinterpret_cast<void*>(err_buffer_size),
			reinterpret_cast<void*>(&err_buffer[0]),
			reinterpret_cast<void*>(max_registers)
		};

		// jit compile (with options)
		cuda_warn( cuModuleLoadDataEx(&module, ptx_source.c_str(), num_options, options, values) );

		if constexpr (verbose)
		   (error ?
			std::cerr:
			std::cout)<< std::left << std::setw(60)
					  << "> PTX JIT info log:" << (error ? x_mark : check_mark) << "\n"
					  << info_buffer.c_str() + "\n< end of info log."s
					  << std::endl;
		
		if (error)
		{
			std::cerr << std::left << std::setw(60)
					  << "> PTX JIT error log:" << x_mark
					  << std::endl;
			throw_err(err_buffer.c_str() + "\n< end of error log."s);
		}
		
		return *this;
	}

	auto& bin_module_load (const std::string& module_path)
	{
		cuda_err(except, cuModuleLoad(&module, module_path.c_str()) );
		
		return *this;
	}

	auto& get_kernel (const std::string& kernel_name)
	{
		if (!module)
			throw_err("Yikes! Make sure to load a module before choosing a kernel.");

		cuda_err(except, cuModuleGetFunction(&kernel, module, kernel_name.c_str()) );

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> Function \"" + kernel_name + "\" found." << check_mark
						  << std::endl;
		
		this->kernel_name = kernel_name;
		
		return *this;
	}

	template <class ...ArgTypes>
	auto& launch (const dim3& grid, const dim3& block, const size_t shared_bytes = 0, ArgTypes&&... args)
	{
		if (!module || !kernel)
			throw_err("Uh-oh. Did you remember to:"
					  "\n\t(1) load a module (.ptx, .cubin, .fatbin) file, and"
					  "\n\t(2) choose a kernel?");
		
		if constexpr(sizeof...(args))
		{
			void* params[] {&args...};
			cuda_err(except, cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, 0, params, NULL) );
		}
		else
			cuda_err(except, cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes, 0, NULL, NULL) );
		
		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
					<< "> Launching kernel:" << check_mark << "\n\t"
					<< kernel_name << "<<<"
					<< "dim3(" << grid.x << "," << grid.y << "," << grid.z << "), dim3(" << block.x << "," << block.y << "," << block.z << "), " << shared_bytes
					<< ">>>" << "("
					<< type_names<std::remove_reference_t<ArgTypes>...>() << ")..."
					<< std::endl;
		
		return *this;
	}

	auto& synchronize ()
	{
		cuda_err(except, cuCtxSynchronize() );

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
					<< "> Synchronize called." << check_mark << "\n\t"
					<< "Blocking CPU until GPU kernel returns..."
					<< std::endl;
		
		return *this;
	}

	template <
	class T = void*,
	class ...ArgTypes,
	class SFINAE = std::enable_if_t< std::is_pointer_v<T> &&
									 (sizeof...(ArgTypes) % 2 == 0) >>
	auto& cu_malloc (T& device_pointer, const size_t& bytes, ArgTypes&&... args)
	{
		auto ptr = reinterpret_cast<CUdeviceptr*>(&device_pointer);
		cuda_err(except, cuMemAlloc(ptr, bytes) );
		to_destroy.push_front(*ptr);

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> " + std::to_string(bytes) + " bytes successfully allocated." << check_mark
						  << std::endl;
		
		if constexpr (sizeof...(args))
			return this->cu_malloc(std::forward<ArgTypes>(args)...);
		else
			return *this;
	}

	template <
	class T,
	class ...ArgTypes,
	class SFINAE = std::enable_if_t< std::is_pointer_v<T> &&
									 (std::is_pointer_v<ArgTypes> && ...) >>
	auto& cu_free (T& device_pointer, ArgTypes&&... args)
	{
		auto ptr = reinterpret_cast<CUdeviceptr>(device_pointer);
		auto search = std::find(cbegin(to_destroy), cend(to_destroy), ptr);
		if (search != cend(to_destroy))
		{
			cuda_err(except, cuMemFree(ptr) );
			if (!error)
				*search = 0; // "erase" the pointer
		}
		else
			warn_err("Oops. The pointer passed (" + std::to_string(ptr) + ") was never allocated any device memory to begin with. (Skipping...)");
		
		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> Device memory pointer successfully freed." << check_mark
						  << std::endl;
		
		if constexpr (sizeof...(args))
			return this->cu_free(std::forward<ArgTypes>(args)...);
		else
			return *this;
	}

	template <class D, class H,
	bool has_data = std::is_pointer_v<decltype(std::declval<H>().data())>,
	class SFINAE = std::enable_if_t< (has_data || std::is_pointer_v<H>) && std::is_pointer_v<D> >>
	auto& memcpy_toDevice (D& dest, H& src, size_t bytes)
	{
		auto device_pointer = reinterpret_cast<CUdeviceptr>(dest);

		if constexpr (has_data)
			cuda_err(except, cuMemcpyHtoD(device_pointer, src.data(), bytes) );
		else
			cuda_err(except, cuMemcpyHtoD(device_pointer, src, bytes) );

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> " + std::to_string(bytes) + " bytes successfully copied (from host) to device." << check_mark
						  << std::endl;
		
		return *this;
	}

	template <class H, class D,
	bool has_data = std::is_pointer_v<decltype(std::declval<H>().data())>,
	class SFINAE = std::enable_if_t< (has_data || std::is_pointer_v<H>) && std::is_pointer_v<D> >>
	auto& memcpy_toHost (H& dest, D& src, size_t bytes)
	{
		auto device_pointer = reinterpret_cast<CUdeviceptr>(src);

		if constexpr (has_data)
			cuda_err(except, cuMemcpyDtoH(dest.data(), device_pointer, bytes) );
		else
			cuda_err(except, cuMemcpyDtoH(dest, device_pointer, bytes) );

		if constexpr (verbose)
			if (!error)
				std::cout << std::left << std::setw(60) << std::setfill(' ')
						  << "> " + std::to_string(bytes) + " bytes successfully copied (from device) to host." << check_mark
						  << std::endl;
		
		return *this;
	}

	operator bool() const
	{
		return !error;
	}
};
#undef cuda_err
#undef cuda_warn
#endif
/* CUDA_CONTEXT_H */



/* Test CudaContext */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <iomanip>	 		// std::setw, std::setprecision, std::left
	#include <iterator> 		// std::ostream_iterator
	#include <algorithm> 		// std::copy
	#include <string> 			// std::string
	#include <vector> 			// std::vector
	#include <random> 			// std::random_device, std::default_random_engine, std::uniform_real_distribution
	#include <cmath> 			// std::abs
	#include "basename.h"

using namespace std;
int main(int argc, char* argv[])
{
try {
	cout << "\n" << endl;
	string module(basename(argv[0]));

	// Test Constructors
	{
		CudaContext<0>();
		CudaContext<0>("cubin/cuda_context.cubin");
		CudaContext<0>("cubin/cuda_context.cubin", "saxpy");
	}

	// Test .cubin kernel (no args, no data transfer)
	{
		cout << setw(61) << setfill('.') << left
			 << "Test .cubin kernel (no args, no data transfer)"
			 << endl;

		cout << "Hello World from the CPU!" << endl;
		CudaContext<1> cuda;
		cuda.module_load("cubin/" + module + ".cubin")
			.get_kernel("kernel")
			.launch(1, 5, 0)
			.synchronize();
		cout << "\n" << endl;

	// Test .ptx kernel (no args, no data transfer), share a previous context
		cout << setw(61) << setfill('.') << left
			 << "Test .ptx kernel (no args, no data transfer),"
			 << "\n"
			 << setw(61)
			 << "Share a previous context"
			 << endl;

		cout << "Hello World from the CPU!\n" << endl;
		cuda.module_load("cubin/" + module + ".ptx")
			.get_kernel("kernel")
			.launch(1, 5, 0)
			.synchronize();
	}
		cout << "\n" << endl;

	{
		// Test .ptx kernel (with args and data transfer)
		cout << setw(61) << setfill('.') << left
			 << "Test .ptx kernel (with args and data transfer)"
			 << endl;
		int N = 1 << 20;
		size_t bytes = N * sizeof(float);
		int block = 1024, grid = (N + block - 1)/block;
		float a = 2.0;
		vector<float> 
			h_x(N), h_y(N), h_result(N);
		float
			*d_x, *d_y, *d_result;
		random_device rd;
		default_random_engine gen(rd());
		uniform_real_distribution<float> distr(0,5);
		cout << "Setup:" << endl;

		for (auto& e : h_x)
			e = distr(gen);
		for (auto& e : h_y)
			e = distr(gen);

		cout << "\t" << "a = " << a << endl;

		if (N <= 10)
		{
			cout << "\t" << "h_x = {";
			copy(cbegin(h_x), cend(h_x), ostream_iterator<float>(cout, ", "));
			cout << "}" << endl;

			cout << "\t" << "h_y = {";
			copy(cbegin(h_y), cend(h_y), ostream_iterator<float>(cout, ", "));
			cout << "}" << endl;

			cout << "\t" << "h_result = {";
			copy(cbegin(h_result), cend(h_result), ostream_iterator<float>(cout, ", "));
			cout << "}" << endl;
			cout << endl;
		}

		CudaContext<1,0> cuda2;
		cuda2.module_load("cubin/" + module + ".ptx")
			 .get_kernel("saxpy")
			 .cu_malloc(d_x,bytes, d_y,bytes, d_result,bytes)
			 .memcpy_toDevice(d_x, h_x, bytes)
			 .memcpy_toDevice(d_y, h_y, bytes)
			 .memcpy_toDevice(d_result, h_result, bytes)
			 .launch(grid, block, 0, a, d_x, d_y, d_result, N)
			 .synchronize()
			 .memcpy_toHost(h_result, d_result, bytes);
		
		if (N <= 10)
		{
			cout << endl;
			cout << "Results:" << endl;

			cout << "\t" << "h_result = {";
			copy(cbegin(h_result), cend(h_result), ostream_iterator<float>(cout, ", "));
			cout << "}" << endl;
		}

		bool fail = false;
		for (int i=0; i<N; i++)
			if (float delta = abs(a*h_x[i] + h_y[i] - h_result[i]); (fail = delta > 1e-4))
				cout << "Difference at index " << i << ": " << delta << endl;

		cout << "\n" << "Verification: " << (fail ? "FAIL" : "PASS") << endl;
	}

	/*
	Potential:
	load/unload multiple modules
	return function handle to kernel
	link module into executable (load with microsoft api)
	async data transfer (pinned memory)
	runtime compilation (source from user input)
	textures/surfaces
	*/
	
} catch (...)
   {return 1;}
	return 0;
}
#endif
/* Test CudaContext */
#endif
/* NVCC */