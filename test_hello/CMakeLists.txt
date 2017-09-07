project (hello)
cmake_minimum_required (VERSION 3.6)

find_package (CUDA REQUIRED)

set (CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}
	"-gencode arch=compute_61,code=sm_61"
	--resource-usage # Show register usage at compilation
	-fmad=false # Disable fast floating math (because less precise)
	)

set (srcs
	"${CMAKE_CURRENT_LIST_DIR}/kernel.cu")

cuda_add_library (${CMAKE_PROJECT_NAME} ${srcs} SHARED)


#[[
	Found out I can just do:
	C:\Users\spenc\OneDrive\Documents\Summer 2017\C++\test_hello> nvcc -arch=sm_61 -Xcompiler /LD kernel.cu -o k.dll
	C:\Users\spenc\OneDrive\Documents\Summer 2017\C++\test_hello> g++ main.cpp -std=c++17 -L. -lk -L"%CUDA_PATH%/lib/x64" -lcudart

	
	-- see cu.bat for more intricate compiler options

	-- according to "nvcc --verbose ..." nvcc already calls vcvarsall.bat with the right target architecture (amd64). So need for this:
	C:\Users\spenc\OneDrive\Documents\Summer 2017\C++\test_hello> vcvarsall.bat amd64

	-- the "-o k.dll" flag is required it seems. g++ makes an executable that expects to find a .dll extension I think.
	   Even when I successfully link to "k.exe" the executable expects to find "k.dll".

	might want to experiment with library inclusion without -L, -l flag -- WORKS
		- try include the .lib file (import library style) perhaps. 	-- WORKS
		  I think this is how the cudart.lib file works.
	experiment with listing all -L args separated with commas 			-- DOESN'T WORK
	try using cudart_static.lib 										-- NO DIFFERENCE? (because import library trick?)
	try multiple .cu files in a library 								-- WORKS (by library, I tried .dll)

	basically:
		- use __declspec(dllexport) / __declspec(dllimport) when communicating with VS compiled libraries
		- use extern "C" to produce compatible mangled names (across compilers) -- requires calling signature to be C compatible
		- no need to worry about calling convention if targeting x64
]]