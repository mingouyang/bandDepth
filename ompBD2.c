#include <R.h>
#include <stdint.h>
#include <omp.h>

#define min(A,B) ((A)<(B) ? (A) : (B))
#define max(A,B) ((A)>(B) ? (A) : (B))
#define Data(i,j) data[(j) * (*row) + (i)] //R uses column-major order

static double *minVec, *maxVec;
static uint64_t *count;

static void countBD(int *row, int *col, double *data){
  unsigned f1, f2, i, j;

  omp_set_num_threads(omp_get_max_threads());
  for (f1 = 0; f1 < (*row)-1; f1++){
    for (f2 = f1+1; f2 < (*row); f2++){
      for (j=0; j<(*col); j++){
	minVec[j] = min(Data(f1, j), Data(f2, j));
	maxVec[j] = max(Data(f1, j), Data(f2, j));
      }

#pragma omp parallel for default(shared) private(i,j)
      for (i=0; i<(*row); i++){
	for (j=0; j<(*col); j++)
	  if (Data(i, j) < minVec[j] || Data(i, j) > maxVec[j])
	    break;
	if (j == (*col))
	  count[i]++;
      }
    }
  }
}

void ompBD2(int *row, int *col, double *data, double *depth){
  unsigned i;

  count = (uint64_t*)malloc(sizeof(uint64_t) * (*row));
  minVec =  (double*)malloc(sizeof(double) * (*col));
  maxVec =  (double*)malloc(sizeof(double) * (*col));

  for (i=0; i<(*row); i++)
    count[i] = 0;
  countBD(row, col, data);
  for (i=0; i<(*row); i++)
    depth[i] = (double)count[i] / ((*row) * (*row - 1.0) / 2.0);

  free(count);
  free(minVec);
  free(maxVec);
}
