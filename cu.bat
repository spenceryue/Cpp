@echo off
nvcc -arch=sm_61 --resource-usage -DNVCC -cubin -x cu "%~1" -o "cubin/%~2.cubin" && ^
rem nvcc -arch=sm_61 --resource-usage -DNVCC -ptx -x cu "%~1" -o "cubin/%~2.ptx" && ^
g++ -O3 -Wall -std=c++17 -I"%CUDA_PATH%/include" -x c++ "%~1" -o "bin/%~2" -L"%CUDA_PATH%/lib/x64" -lcuda && ^
if not "%~3" == "" "bin\%~2"