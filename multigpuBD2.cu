#include <R.h>
#include <stdint.h>

#define min(A,B)  ((A)<(B) ? (A) : (B))
#define max(A,B)  ((A)>(B) ? (A) : (B))
#define Data(i,j) data[(j) * (*row) + (i)] //R uses column-major order
#define Func(i,j) func[(i) * m + (j)]

#define numThread   256
#define MaxN      65536
#define MaxCol      256

static unsigned *count, *tmpCount, *f1, *f2, **gpuCount, **gpuF1, **gpuF2;
static float *func, **gpuFunc;

__global__
void kernel(float *func, unsigned *count, unsigned n, unsigned m,
	    unsigned *f1, unsigned *f2){
  __shared__ float minVector[MaxCol];
  __shared__ float maxVector[MaxCol];
  unsigned myFunc, i, j;
  float funcValue;

  if (threadIdx.x < m){
    funcValue = Func(f1[blockIdx.x], threadIdx.x); //func 1
    minVector[threadIdx.x] = funcValue;
    maxVector[threadIdx.x] = funcValue;
    funcValue = Func(f2[blockIdx.x], threadIdx.x); //func 2
    minVector[threadIdx.x] = min(minVector[threadIdx.x], funcValue);
    maxVector[threadIdx.x] = max(maxVector[threadIdx.x], funcValue);
  }
  __syncthreads();

  for (i=0; i<n; i += blockDim.x){
    myFunc = i + threadIdx.x;
    if (myFunc < n){
      for (j=0; j<m; j++){
	funcValue = Func(myFunc, j);
	if (funcValue < minVector[j] || funcValue > maxVector[j])
	  break;
      }
      if (j == m)
	atomicAdd(count + myFunc, 1);
    }
  }
}

extern "C"
void multigpuBD2(int *row, int *col, double *data, double *depth){
  unsigned n, m, chunk, size;
  uint64_t i, j, k, numPairs;
  int numGPU;

  n = *row;
  m = *col;
  cudaGetDeviceCount(&numGPU);
  if (n > MaxN){
    fprintf(stderr, "number of rows cannot be more than %u\n", MaxN);
    exit(1);
  }
  if (m > MaxCol){
    fprintf(stderr, "number of columns cannot be more than %u\n", MaxCol);
    exit(1);
  }
  if (numGPU < 2){
    fprintf(stderr, "need more than 1 GPU\n");
    exit(1);
  }

  count    = (unsigned*)malloc(sizeof(unsigned) * n);
  tmpCount = (unsigned*)malloc(sizeof(unsigned) * n);
  func     =    (float*)malloc(sizeof(float) * n * m);
  for (i=0; i<n; i++){
    count[i] = 0;
    for (j=0; j<m; j++)
      Func(i, j) = Data(i, j);
    //data: column major, double
    //func: row major, float
  }
  numPairs = (uint64_t)n * (n-1) / 2;
  f1 = (unsigned*)malloc(sizeof(unsigned) * numPairs);
  f2 = (unsigned*)malloc(sizeof(unsigned) * numPairs);
  for (i=0, k=0; i<n; i++)
    for (j=i+1; j<n; j++)
      f1[k] = i, f2[k++] = j;

  chunk = (numPairs + numGPU - 1) / numGPU;
  gpuCount = (unsigned**)malloc(numGPU * sizeof(unsigned*));
  gpuF1    = (unsigned**)malloc(numGPU * sizeof(unsigned*));
  gpuF2    = (unsigned**)malloc(numGPU * sizeof(unsigned*));
  gpuFunc  =    (float**)malloc(numGPU * sizeof(float*));
  for (i=0; i<numGPU; i++){
    cudaSetDevice(i);
    cudaMalloc((void**)&gpuCount[i], sizeof(unsigned) * n);
    cudaMalloc((void**)&gpuFunc[i],  sizeof(float) * n * m);
    cudaMalloc((void**)&gpuF1[i],    sizeof(unsigned) * chunk);
    cudaMalloc((void**)&gpuF2[i],    sizeof(unsigned) * chunk);
    size = (i == numGPU - 1) ? (numPairs - i * chunk) : chunk;
    cudaMemcpy(gpuCount[i], count, sizeof(unsigned) * n,
	       cudaMemcpyHostToDevice);
    cudaMemcpy(gpuFunc[i], func, sizeof(float) * n * m,
	       cudaMemcpyHostToDevice);
    cudaMemcpy(gpuF1[i], &f1[i*chunk], sizeof(unsigned) * size,
	       cudaMemcpyHostToDevice);
    cudaMemcpy(gpuF2[i], &f2[i*chunk], sizeof(unsigned) * size,
	       cudaMemcpyHostToDevice);
    kernel<<<size, numThread>>>(gpuFunc[i], gpuCount[i], n, m,
				gpuF1[i], gpuF2[i]);
  }
  for (i=0; i<numGPU; i++){
    cudaSetDevice(i);
    cudaThreadSynchronize();
    cudaMemcpy(tmpCount, gpuCount[i], sizeof(unsigned) * n,
	       cudaMemcpyDeviceToHost);
    for (j=0; j<n; j++)
      count[j] += tmpCount[j];
    cudaFree(gpuCount[i]);
    cudaFree(gpuFunc[i]);
    cudaFree(gpuF1[i]);
    cudaFree(gpuF2[i]);
  }
  for (i=0; i<n; i++)
    depth[i] = (double)count[i] / (n * (n - 1.0) / 2.0);

  free(count);
  free(tmpCount);
  free(func);
  free(f1);
  free(f2);
  free(gpuCount);
  free(gpuFunc);
  free(gpuF1);
  free(gpuF2);
}
