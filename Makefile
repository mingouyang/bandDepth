gpuBD.so: gpuBD2.cu gpuBD3.cu multigpuBD2.cu multigpuBD3.cu
	nvcc -O3 -I/usr/include/R --shared -Xcompiler -fPIC -o gpuBD.so gpuBD2.cu gpuBD3.cu multigpuBD2.cu multigpuBD3.cu

ompBD.so: ompBD2.c ompBD3.c askewBD2.c lowDimBD2.c
	icc -shared -fPIC -I/usr/include/R -qopenmp -o ompBD.so ompBD2.c ompBD3.c askewBD2.c lowDimBD2.c
