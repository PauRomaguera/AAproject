#include <stdio.h>
#include <stdlib.h>
#define N 1500
#define REAL double
#define DQSZ 2  // Base case size (2x2)

void GJ_Base(REAL* input, REAL* inverse, int SZ) {
int i, j, k;
  REAL pivot, pivot2, sum;
  for (i = 0; i < SZ; i++) 
  {
    pivot = input[i*N+i];
    for (j = 0; j < N; j++)
    {
      input[i*N+j] /= pivot;
      inverse[i*N+j] /= pivot;
    }
    for (j = 0; j < SZ; j++)
      if (i != j) 
      {
        pivot2 = input[j*N+i];
        for (k = 0; k < SZ; k++)
          input[j * N + k] -= input[i * N + k] * pivot2;
        for (k = 0; k < SZ; k++)
          inverse[j * N + k] -= inverse[i * N + k] * pivot2;
      }
  }
}


void GaussJordan_DQ(REAL* input, REAL* inverse, int SZ) {
    if (SZ <= DQSZ) {
        // Base case: use Gauss-Jordan elimination for small matrices
        GJ_Base(input, inverse, SZ);
        return;
    }

    // SZ is reduced by half
    SZ = SZ / 2;

    // Divide matrix into 4 blocks and recurse
    GaussJordan_DQ(input,          inverse,          SZ);  // Top-left block (A11)
    GaussJordan_DQ(input + SZ,     inverse + SZ,     SZ);  // Top-right block (A12)
    GaussJordan_DQ(input + SZ * N, inverse + SZ * N, SZ);  // Bottom-left block (A21)
    GaussJordan_DQ(input + SZ * (N + 1), inverse + SZ * (N + 1), SZ);  // Bottom-right block (A22)
}


int main() 
{
  printf("Matrix size: %dx%d\n", N, N);
  double sum = 0;
  REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
  REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));
  int i, j, k;
    // initializes matrix with pseudo-random numbers (always the same)
  srand(0u);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      input[i*N+j] = ((REAL)rand()/RAND_MAX);

  // initializes identity matrix
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (i == j)
        inverse[i*N+j] = 1.0f;
      else
        inverse[i*N+j] = 0.0f;
        
  GaussJordan_DQ(input, inverse, N);
          // computes checksum
  sum = 0.0f;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      sum += inverse[i * N + j];
free(input);
free(inverse);
  printf("Checksum: %.8f\n", sum);
  return 0;
}

  