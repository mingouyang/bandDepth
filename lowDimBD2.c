#include <R.h>
#include <stdint.h>
#include <x86intrin.h>
#include <omp.h>

#define min(A,B)    ((A) < (B) ? (A) : (B))
#define max(A,B)    ((A) > (B) ? (A) : (B))
#define Data(i,j)     data[(j) * n + (i)] //R uses column-major order
#define Sorted(i,j) sorted[(j) * n + (i)]
#define Idx(i,j)       idx[(j) * n + (i)]
#define Rank(i,j)     rank[(j) * n + (i)]
#define Ticks(t,j)   ticks[(j) * numTick + (t)]

#define MaxSample  64
#define TICK       7
#define OneTick    128        //2^TICK functions per tick
#define WORD       6          //uint64_t, 2^6
#define OneWord    64
#define MASK       0x0000003F //6 1's
#define ONES       0xFFFFFFFFFFFFFFFFull //64 1's

struct node {
  double val;
  unsigned idx;
};
typedef struct node node;
static node *toSort;

static unsigned n, m, numThread, *idx, *rank, numTick, vecLen;
static uint64_t *count;
static double *sorted;
static uint64_t **ticks;  //numTick * m
static unsigned threeAs;  //3^m
static uint64_t ***ltVec, ***eqVec, ***gtVec;
static int8_t **first, **second;

static uint64_t oneBit[64];
//oneBit[0] 0001
//oneBit[1] 0010
//oneBit[2] 0100
//oneBit[3] 1000
//etc
//oneBit[63]

static int cmpNode(const void *a, const void *b) {
  double x, y;

  x = ((node*) a)->val;
  y = ((node*) b)->val;
  //descending order
  if (x > y) return -1;
  if (x < y) return 1;
  return 0;
}

static void sortFunc(double *data) {
  unsigned i, j;

  for (j = 0; j < m; j++) {
    for (i = 0; i < n; i++) {
      toSort[i].val = Data(i, j);
      toSort[i].idx = i;
    }
    qsort((void*)toSort, n, sizeof(node), cmpNode);
    for (i = 0; i < n; i++) {
      Idx(i, j) = toSort[i].idx;    //which func has rank i at sample j
      Sorted(i, j) = toSort[i].val; //func value at rank i, sample j
      Rank(toSort[i].idx, j) = i;   //rank of function i at sample j
    }
  }
}

static void initTick(void) {
  unsigned i, j, k, t, id, w, o;

  for (j = 0; j < m; j++) {
    for (k = 0; k < vecLen; k++)
      Ticks(0, j)[k] = 0;
    i = 0;
    for (t = 1; t < numTick; t++) {
      for (k = 0; k < vecLen; k++)
	Ticks(t, j) [k] = Ticks(t - 1, j) [k];
      while (i < t * OneTick) {
	id = Idx(i, j), w = id >> WORD, o = id & MASK;
	Ticks(t, j) [w] |= oneBit[o];
	i++;
      }
    }
  }
}

inline uint64_t countCount(unsigned tid) {
  unsigned i, j, k, o, r, s, numAbove, numBelow, numEqual;
  uint64_t cnt, above, below, equal;

  numEqual = 0;
  for (k = 0; k < vecLen; k++) {
    equal = ONES;
    for (s = 0; s < m; s++)
      equal &= eqVec[tid] [s] [k];
    numEqual += _mm_popcnt_u64(equal);
  }
  cnt = (n - numEqual) * numEqual + numEqual * (numEqual - 1) / 2;
  for (i = 1; i < threeAs; i++) {
    o = i;
    for (s = 0; s < m; s++) {
      r = o % 3, o = o / 3;
      first[tid] [s] = (r == 0) ? 0 : (r == 1) ? -1 : 1;
    }
    for (j = i + 1; j < threeAs; j++) {
      o = j;
      for (s = 0; s < m; s++) {
	r = o % 3, o = o / 3;
	second[tid] [s] = (r == 0) ? 0 : (r == 1) ? -1 : 1;
      }
      //-1, 0, 1 for <, =, >
      //for a properly formed band, first[s] * second[s] <= 0, for all s
      for (s = 0; s < m; s++)
	if (first[tid] [s] * second[tid] [s] == 1)
	  break;
      if (s < m) continue;
      numAbove = numBelow = 0;
      for (k = 0; k < vecLen; k++) {
	above = below = ONES;
	for (s = 0; s < m; s++) {
	  above &= (first[tid] [s] == -1) ? ltVec[tid] [s] [k] :
	    (first[tid] [s] == 0) ? eqVec[tid] [s] [k] : gtVec[tid] [s] [k];
	  below &= (second[tid] [s] == -1) ? ltVec[tid] [s] [k] :
	    (second[tid] [s] == 0) ? eqVec[tid] [s] [k] : gtVec[tid] [s] [k];
	}
	numAbove += _mm_popcnt_u64(above);
	numBelow += _mm_popcnt_u64(below);
      }
      cnt += numAbove * numBelow;
    }
  }
  return cnt;
}

static void calcDepth(void) {
  //f func, r rank, s sample, t tick, w word, o offset
  unsigned f, k, r, s, t, id, w, o, threadID;
  int i;

#pragma omp parallel for private(i,k,r,s,t,id,w,o,threadID)
  for (f = 0; f < n; f++) {
    threadID = omp_get_thread_num();
    for (s = 0; s < m; s++) {
      r = Rank(f, s);
      t = r >> TICK;
      for (k = 0; k < vecLen; k++)
	gtVec[threadID] [s] [k] = Ticks(t, s) [k];
      i = t << TICK;
      while (i < n && Sorted(i, s) >= Sorted(r, s)) {
	id = Idx(i, s), w = id >> WORD, o = id & MASK;
	gtVec[threadID] [s] [w] |= oneBit[o];
	i++;
      }
      for (k = 0; k < vecLen; k++)
	ltVec[threadID] [s] [k] = ~ gtVec[threadID] [s] [k];
      while ((i - 1) >= 0 && Sorted(i - 1, s) <= Sorted(r, s)) {
	i--;
	id = Idx(i, s), w = id >> WORD, o = id & MASK;
	ltVec[threadID] [s] [w] |= oneBit[o];
      }
      for (k = 0; k < vecLen; k++) {
	eqVec[threadID] [s] [k] = gtVec[threadID] [s] [k] &
	  ltVec[threadID] [s] [k];
	gtVec[threadID] [s] [k] &= ~ eqVec[threadID] [s] [k];
	ltVec[threadID] [s] [k] &= ~ eqVec[threadID] [s] [k];
      }
      //padded functions, id >= n, are less than all functions
      for (id = n; id < (vecLen << WORD); id++) {
	w = id >> WORD, o = id & MASK;
	ltVec[threadID] [s] [w] &= ~ oneBit[o];
      }
    }
    count[f] = countCount(threadID);
  }
}

void lowDimBD2(int *row, int *col, double *data, double *depth) {
  unsigned i, j;

  n = *row;
  m = *col;
  if (n <= OneTick) {
    fprintf(stderr, "minimum %u rows\n", OneTick + 1);
    exit(1);
  }
  if (m > MaxSample) {
    fprintf(stderr, "maximum %u columns\n", MaxSample);
    exit(1);
  }
  numThread = omp_get_max_threads();
  omp_set_num_threads(numThread);

  count  = (uint64_t*) malloc(sizeof(uint64_t) * n);
  toSort = (node*)     malloc(sizeof(node)     * n);
  sorted = (double*)   malloc(sizeof(double)   * n * m);
  idx    = (unsigned*) malloc(sizeof(unsigned) * n * m);
  rank   = (unsigned*) malloc(sizeof(unsigned) * n * m);

  oneBit[0] = 1;
  for (i = 1; i < 64; i++)
    oneBit[i] = oneBit[i - 1] << 1;
  for (i = 0, threeAs = 1; i < m; i++)
    threeAs *= 3;

  //one tick mark every 128 functions
  numTick = (n >> TICK) + 1;
  ticks = (uint64_t**)malloc(sizeof(uint64_t*) * numTick * m);
  //num of 64-bit words
  vecLen = (n >> WORD) + 1;
  for (i = 0; i < numTick * m; i++)
    ticks[i] = (uint64_t*)malloc(sizeof(uint64_t) * vecLen);

  gtVec = (uint64_t***)malloc(sizeof(uint64_t**) * numThread);
  ltVec = (uint64_t***)malloc(sizeof(uint64_t**) * numThread);
  eqVec = (uint64_t***)malloc(sizeof(uint64_t**) * numThread);
  for (i = 0; i < numThread; i++) {
    gtVec[i] = (uint64_t**)malloc(sizeof(uint64_t*) * m);
    ltVec[i] = (uint64_t**)malloc(sizeof(uint64_t*) * m);
    eqVec[i] = (uint64_t**)malloc(sizeof(uint64_t*) * m);
    for (j = 0; j < m; j++) {
      gtVec[i] [j] = (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
      ltVec[i] [j] = (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
      eqVec[i] [j] = (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
    }
  }
  first  = (int8_t**)malloc(sizeof(int8_t*) * numThread);
  second = (int8_t**)malloc(sizeof(int8_t*) * numThread);
  for (i = 0; i < numThread; i++) {
    first[i]  = (int8_t*)malloc(sizeof(int8_t) * MaxSample);
    second[i] = (int8_t*)malloc(sizeof(int8_t) * MaxSample);
  }

  sortFunc(data);
  initTick();
  calcDepth();
  for (i = 0; i < n; i++)
    depth[i] = (double)count[i] / (n * (n - 1.0) / 2.0);

  free(count);
  free(toSort);
  free(sorted);
  free(idx);
  free(rank);

  for (i = 0; i < numTick*m; i++)
    free(ticks[i]);
  free(ticks);

  for (i = 0; i < numThread; i++) {
    for (j=0; j<m; j++){
      free(gtVec[i] [j]);
      free(ltVec[i] [j]);
      free(eqVec[i] [j]);
    }
    free(gtVec[i]);
    free(ltVec[i]);
    free(eqVec[i]);
  }
  free(gtVec);
  free(ltVec);
  free(eqVec);

  for (i = 0; i < numThread; i++) {
    free(first[i]);
    free(second[i]);
  }
  free(first);
  free(second);
}
