#include <R.h>

#define min(A,B)  ((A) < (B) ? (A) : (B))
#define max(A,B)  ((A) > (B) ? (A) : (B))
#define Data(i,j) data[(j) * (*row) + (i)] //R uses column-major order
#define Func(i,j) func[(i) * m + (j)]

#define numThread   256
#define MaxN      65536
#define MaxCol      256

static unsigned *count, *gpuCount;
static float *func, *gpuFunc;

__global__ void kernel(float *func, unsigned *count, unsigned n, unsigned m) {
  __shared__ float minVector[MaxCol];
  __shared__ float maxVector[MaxCol];
  unsigned myFunc, i, j;
  float funcValue;

  //blockIdx.x is the index of func 1
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

extern "C"
void gpuBD2(int *row, int *col, double *data, double *depth) {
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

  count = (unsigned*)malloc(sizeof(unsigned) * n);
  func = (float*)malloc(sizeof(float) * n * m);
  cudaSetDevice(0);
  cudaMalloc((void**)&gpuCount, sizeof(unsigned) * n);
  cudaMalloc((void**)&gpuFunc, sizeof(float) * n * m);

  for (i = 0; i < n; i++) {
    count[i] = 0;
    for (j = 0; j < m; j++)
      Func(i, j) = Data(i, j);
    //data: column major, double
    //func: row major, float
  }

  cudaMemcpy(gpuFunc, func, sizeof(float) * n * m, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuCount, count, sizeof(unsigned) * n, cudaMemcpyHostToDevice);
  dim3 grid(n, n - 1);
  kernel<<<grid, numThread>>>(gpuFunc, gpuCount, n, m);
  cudaThreadSynchronize();
  cudaMemcpy(count, gpuCount, sizeof(unsigned) * n, cudaMemcpyDeviceToHost);

  for (i = 0; i < n; i++)
    depth[i] = (double)count[i] / (n * (n - 1.0) / 2.0);

  free(count);
  free(func);
  cudaFree(gpuCount);
  cudaFree(gpuFunc);
}
