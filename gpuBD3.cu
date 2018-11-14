#include <R.h>

#define min(A,B)  ((A) < (B) ? (A) : (B))
#define max(A,B)  ((A) > (B) ? (A) : (B))
#define Data(i,j) data[(j) * (*row) + (i)] //R uses column-major order
#define Func(i,j) func[(i) * m + (j)]

#define numThread   256
#define MaxN      65536
#define MaxCol      256

static unsigned *count2, *count3, *gpuCount2, *gpuCount3;
static float *func, *gpuFunc;

__global__ void kernel2(float *func, unsigned *count, unsigned n, unsigned m) {
  __shared__ float minVector[MaxCol];
  __shared__ float maxVector[MaxCol];
  unsigned myFunc, i, j;
  float funcValue;

  //blockIdx.x     is the index of func 1
  //blockIdx.y + 1 is the index of func 2
  if (blockIdx.y + 1 <= blockIdx.x)
    return;
  if (threadIdx.x < m) {
    funcValue = Func(blockIdx.x, threadIdx.x); //func 1
    minVector[threadIdx.x] = funcValue;
    maxVector[threadIdx.x] = funcValue;
    funcValue = Func(blockIdx.y + 1, threadIdx.x); //func 2
    minVector[threadIdx.x] = min(minVector[threadIdx.x], funcValue);
    maxVector[threadIdx.x] = max(maxVector[threadIdx.x], funcValue);
  }
  __syncthreads();

  for (i = 0; i < n; i += blockDim.x) {
    myFunc = i + threadIdx.x;
    if (myFunc < n) {
      for (j = 0; j < m; j++) {
	funcValue = Func(myFunc, j);
	if (funcValue < minVector[j] || funcValue > maxVector[j])
	  break;
      }
      if (j == m)
	atomicAdd(count + myFunc, 1);
    }
  }
}

__global__ void kernel3(float *func, unsigned *count, unsigned n, unsigned m) {
  __shared__ float minVector[MaxCol];
  __shared__ float maxVector[MaxCol];
  unsigned myFunc, i, j;
  float funcValue;

  //blockIdx.x     is the index of func 1
  //blockIdx.y + 1 is the index of func 2
  //blockIdx.z + 2 is the index of func 3
  if (blockIdx.y + 1 <= blockIdx.x || blockIdx.z + 1 <= blockIdx.y)
    return;
  if (threadIdx.x < m) {
    funcValue = Func(blockIdx.x, threadIdx.x); //func 1
    minVector[threadIdx.x] = funcValue;
    maxVector[threadIdx.x] = funcValue;
    funcValue = Func(blockIdx.y + 1, threadIdx.x); //func 2
    minVector[threadIdx.x] = min(minVector[threadIdx.x], funcValue);
    maxVector[threadIdx.x] = max(maxVector[threadIdx.x], funcValue);
    funcValue = Func(blockIdx.z + 2, threadIdx.x); //func 3
    minVector[threadIdx.x] = min(minVector[threadIdx.x], funcValue);
    maxVector[threadIdx.x] = max(maxVector[threadIdx.x], funcValue);
  }
  __syncthreads();

  for (i = 0; i < n; i += blockDim.x) {
    myFunc = i + threadIdx.x;
    if (myFunc < n) {
      for (j = 0; j < m; j++) {
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
void gpuBD3(int *row, int *col, double *data, double *depth) {
  unsigned i, j, n, m;

  n = *row;
  m = *col;
  if (n > MaxN) {
    fprintf(stderr, "number of rows cannot be more than %u\n", MaxN);
    exit(1);
  }
  if (m > MaxCol) {
    fprintf(stderr, "number of columns cannot be more than %u\n", MaxCol);
    exit(1);
  }

  count2 = (unsigned*)malloc(sizeof(unsigned) * n);
  count3 = (unsigned*)malloc(sizeof(unsigned) * n);
  func = (float*)malloc(sizeof(float) * n * m);
  cudaSetDevice(0);
  cudaMalloc((void**)&gpuCount2, sizeof(unsigned) * n);
  cudaMalloc((void**)&gpuCount3, sizeof(unsigned) * n);
  cudaMalloc((void**)&gpuFunc, sizeof(float) * n * m);

  for (i = 0; i < n; i++) {
    count2[i] = count3[i] = 0;
    for (j = 0; j < m; j++)
      Func(i, j) = Data(i, j);
    //data: column major, double
    //func: row major, float
  }

  cudaMemcpy(gpuFunc, func, sizeof(float) * n * m, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuCount2, count2, sizeof(unsigned) * n, cudaMemcpyHostToDevice);
  dim3 grid2(n, n - 1);
  kernel2<<<grid2, numThread>>>(gpuFunc, gpuCount2, n, m);
  cudaThreadSynchronize();
  cudaMemcpy(count2, gpuCount2, sizeof(unsigned) * n, cudaMemcpyDeviceToHost);

  cudaMemcpy(gpuCount3, count3, sizeof(unsigned) * n, cudaMemcpyHostToDevice);
  dim3 grid3(n, n - 1, n - 2);
  kernel3<<<grid3, numThread>>>(gpuFunc, gpuCount3, n, m);
  cudaThreadSynchronize();
  cudaMemcpy(count3, gpuCount3, sizeof(unsigned) * n, cudaMemcpyDeviceToHost);

  for (i = 0; i < n; i++)
    depth[i] = (double)count2[i] / (n * (n - 1.0) / 2.0) +
      (double)count3[i] / (n * (n - 1.0) * (n - 2.0) / 6.0);

  free(count2);
  free(count3);
  free(func);
  cudaFree(gpuCount2);
  cudaFree(gpuCount3);
  cudaFree(gpuFunc);
}
