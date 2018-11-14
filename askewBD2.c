#include <R.h>
#include <stdint.h>
#include <x86intrin.h>
#include <omp.h>

#define min(A,B)    ((A)<(B) ? (A) : (B))
#define max(A,B)    ((A)>(B) ? (A) : (B))
#define Data(i,j)     data[(j) * n + (i)] //R uses column-major order
#define Sorted(i,j) sorted[(j) * n + (i)]
#define Idx(i,j)       idx[(j) * n + (i)]
#define Rank(i,j)     rank[(j) * n + (i)]
#define Ticks(t,j)   ticks[(j) * numTick + (t)]

#define TICK    7
#define OneTick 128        //2^TICK functions per tick
#define WORD    6          //uint64_t, 2^6
#define OneWord 64
#define MASK    0x0000003F //6 1's
#define ONES    0xFFFFFFFFFFFFFFFFull //64 1's

struct node {
  double val;
  unsigned idx;
};
typedef struct node node;
static node *toSort;

static unsigned n, m, numThread, *idx, *rank, numTick, vecLen;
static uint64_t *count;
static double   *sorted;
static uint64_t **ticks;   //numTick * m
static unsigned twoAs;     //2^m
static uint64_t ***bitVec; //twoAs
static uint64_t **EQ, **GE, **LE;
static unsigned **cnt;

static uint64_t oneBit[64];
//oneBit[0] 0001
//oneBit[1] 0010
//oneBit[2] 0100
//oneBit[3] 1000

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
    qsort((void *) toSort, n, sizeof(node), cmpNode);
    for (i = 0; i < n; i++) {
      Idx(i, j)    = toSort[i].idx; //which row has rank i at col j
      Sorted(i, j) = toSort[i].val; //func value at rank i, col j
      Rank(toSort[i].idx, j) = i;   //rank of row i at col j
    }
  }
}

static void initTick(void) {
  unsigned i, j, k, t, id, w, o;

  for (j = 0; j < m; j++) {
    for (k = 0; k < vecLen; k++)
      Ticks(0, j) [k] = 0;
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

static void calcDepth(void) {
  //f func, r rank, c col, t tick, v vector, w word, o offset
  unsigned f, k, r, c, t, v, id, w, o, threadID, cntEQ;
  int i;

#pragma omp parallel for private(i,k,r,c,t,id,w,o,threadID,cntEQ)
  for (f = 0; f < n; f++) {
    threadID = omp_get_thread_num();
    for (v = 0; v < twoAs; v++)
      for (k = 0; k < vecLen; k++)
	bitVec[threadID] [v] [k] = ONES;
    for (k = 0; k < vecLen; k++)
      EQ[threadID] [k] = ONES;
    for (c = 0; c < m; c++) {
      r = Rank(f, c);
      t = r >> TICK;
      for (k = 0; k < vecLen; k++)
	GE[threadID] [k] = Ticks(t, c) [k];
      i = t << TICK;
      while (i < n && Sorted(i, c) >= Sorted(r, c)) {
	id = Idx(i, c), w = id >> WORD, o = id & MASK;
	GE[threadID] [w] |= oneBit[o];
	i++;
      }
      for (k = 0; k < vecLen; k++)
	LE[threadID] [k] = ~ GE[threadID] [k];
      i--;
      while (i >= 0 && Sorted(i, c) <= Sorted(r, c)) {
	id = Idx(i, c), w = id >> WORD, o = id & MASK;
	LE[threadID] [w] |= oneBit[o];
	i--;
      }
      for (v = 0; v < twoAs; v++)
	for (k = 0; k < vecLen; k++)
	  bitVec[threadID] [v] [k] &= ((v & oneBit[c]) ? GE[threadID] [k] :
				       LE[threadID] [k]);
      for (k = 0; k < vecLen; k++)
	EQ[threadID] [k] &= GE[threadID] [k] & LE[threadID] [k];
    }
    cntEQ = 0;
    for (k = 0; k < vecLen; k++)
      cntEQ += _mm_popcnt_u64( EQ[threadID] [k] );
    for (v = 0; v < twoAs; v++) {
      cnt[threadID] [v] = 0;
      for (k = 0; k < vecLen; k++)
	cnt[threadID] [v] += _mm_popcnt_u64( bitVec[threadID] [v] [k] );
      cnt[threadID] [v] -= cntEQ;
    }
    //fictitious functions with indices >= n, are less than all functions
    cnt[threadID] [0] -= ((vecLen << WORD) - n);
    count[f] = (n - cntEQ) * cntEQ + cntEQ * (cntEQ - 1) / 2;
    for (v = 0; v < twoAs >> 1; v++)
      count[f] += cnt[threadID] [v] * cnt[threadID] [twoAs - 1 - v];
  }
}

void askewBD2(int *row, int *col, double *data, double *depth) {
  unsigned i, j;

  n = *row;
  m = *col;
  if (n <= OneTick) {
    fprintf(stderr, "minimum %u rows\n", OneTick + 1);
    exit(1);
  }

  numThread = omp_get_max_threads();
  omp_set_num_threads(numThread);

  count  = (uint64_t*)malloc(sizeof(uint64_t) * n);
  toSort = (node*)    malloc(sizeof(node)     * n);
  sorted = (double*)  malloc(sizeof(double)   * n * m);
  idx    = (unsigned*)malloc(sizeof(unsigned) * n * m);
  rank   = (unsigned*)malloc(sizeof(unsigned) * n * m);

  oneBit[0] = 1;
  for (i = 1; i < 64; i++)
    oneBit[i] = oneBit[i - 1] << 1;
  twoAs = 1 << m;

  //one tick mark every 128 functions
  numTick = (n >> TICK) + 1;
  ticks = (uint64_t**)malloc(sizeof(uint64_t*) * numTick * m);
  //num of 64-bit words
  vecLen = (n >> WORD) + 1;
  for (i = 0; i < numTick * m; i++)
    ticks[i] = (uint64_t*)malloc(sizeof(uint64_t) * vecLen);

  bitVec = (uint64_t***)malloc(sizeof(uint64_t**) * numThread);
  for (i = 0; i < numThread; i++) {
    bitVec[i] = (uint64_t**)malloc(sizeof(uint64_t*) * twoAs);
    for (j = 0; j < twoAs; j++)
      bitVec[i] [j] = (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
  }
  GE =  (uint64_t**)malloc(sizeof(uint64_t*) * numThread);
  LE =  (uint64_t**)malloc(sizeof(uint64_t*) * numThread);
  EQ =  (uint64_t**)malloc(sizeof(uint64_t*) * numThread);
  cnt = (unsigned**)malloc(sizeof(unsigned*) * numThread);
  for (i = 0; i < numThread; i++) {
    GE[i] =  (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
    LE[i] =  (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
    EQ[i] =  (uint64_t*)malloc(sizeof(uint64_t) * vecLen);
    cnt[i] = (unsigned*)malloc(sizeof(unsigned) * twoAs);
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
  for (i = 0; i < numTick * m; i++)
    free(ticks[i]);
  free(ticks);
  for (i = 0; i < numThread; i++) {
    for (j = 0; j < twoAs; j++)
      free(bitVec[i] [j]);
    free(bitVec[i]);
  }
  free(bitVec);
  for (i = 0; i < numThread; i++) {
    free(GE[i]);
    free(LE[i]);
    free(EQ[i]);
    free(cnt[i]);
  }
  free(GE);
  free(LE);
  free(EQ);
  free(cnt);
}
