#include <R.h>
#include <stdint.h>
#include <omp.h>

#define min(A,B) ((A)<(B) ? (A) : (B))
#define max(A,B) ((A)>(B) ? (A) : (B))
#define Data(i,j) data[(j) * (*row) + (i)] //R uses column-major order

static double *minVec, *maxVec, *twoMin, *twoMax;
static uint64_t *count2, *count3;

static void countBD(int *row, int *col, double *data){
  unsigned f1, f2, f3, i, j;

  omp_set_num_threads(omp_get_max_threads());
  for (f1 = 0; f1 < (*row)-1; f1++){
    for (f2 = f1+1; f2 < (*row); f2++){
      for (j=0; j<(*col); j++){
	twoMin[j] = min(Data(f1, j), Data(f2, j));
	twoMax[j] = max(Data(f1, j), Data(f2, j));
      }

#pragma omp parallel for default(shared) private(i,j)
      for (i=0; i<(*row); i++){
	for (j=0; j<(*col); j++)
	  if (Data(i, j) < twoMin[j] || Data(i, j) > twoMax[j])
	    break;
	if (j == (*col))
	  count2[i]++;
      }

      for (f3=f2+1; f3<(*row); f3++){
	for (j=0; j<(*col); j++){
	  minVec[j] = min(twoMin[j], Data(f3,j));
	  maxVec[j] = max(twoMax[j], Data(f3,j));
	}

#pragma omp parallel for private(i,j)
	for (i=0; i<(*row); i++){
	  for (j=0; j<(*col); j++)
	    if (Data(i,j) < minVec[j] || Data(i,j) > maxVec[j])
	      break;
	  if (j == (*col))
	    count3[i]++;
	}
      }
    }
  }
}

void ompBD3(int *row, int *col, double *data, double *depth){
  unsigned i;

  count2 = (uint64_t*)malloc(sizeof(uint64_t) * (*row));
  count3 = (uint64_t*)malloc(sizeof(uint64_t) * (*row));
  twoMin =   (double*)malloc(sizeof(double) * (*col));
  twoMax =   (double*)malloc(sizeof(double) * (*col));
  minVec =   (double*)malloc(sizeof(double) * (*col));
  maxVec =   (double*)malloc(sizeof(double) * (*col));

  for (i=0; i<(*row); i++)
    count2[i] = count3[i] = 0;
  countBD(row, col, data);
  for (i=0; i<(*row); i++)
    depth[i] = (double)count2[i] / ((*row) * (*row - 1.0) / 2.0) +
      (double)count3[i] / ((*row) * (*row - 1.0) * (*row - 2.0) / 6.0);

  free(count2);
  free(count3);
  free(twoMin);
  free(twoMax);
  free(minVec);
  free(maxVec);
}
