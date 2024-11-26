#include <stdio.h>
#include <stdlib.h>

#define REAL double
#define N 1500

 
int main() 
{
  printf("Matrix size: %dx%d\n", N, N);

  REAL *input = (REAL *)malloc(N * N * sizeof(REAL));
  REAL *inverse = (REAL *)malloc(N * N * sizeof(REAL));
  REAL pivot, pivot2, sum;
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

  // computes the inverse matrix
  for (i = 0; i < N; i++) 
  {
    pivot = input[i*N+i];
    for (j = 0; j < N; j++)
    {
      input[i*N+j] /= pivot;
      inverse[i*N+j] /= pivot;
    }
    for (j = 0; j < N; j++)
      if (i != j) 
      {
        pivot2 = input[j*N+i];
        for (k = 0; k < N; k++)
          input[j * N + k] -= input[i * N + k] * pivot2;
        for (k = 0; k < N; k++)
          inverse[j * N + k] -= inverse[i * N + k] * pivot2;
      }
  }
		
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
